#pragma once

// RenderFeature: the extension point of the renderer.
//
// A feature is one self-contained rendering technique (path tracer, a denoiser, a new GI
// experiment, ...). It lives in src/features/<name>/ with its own C++, shaders and tests, depends
// only on core / rhi / rdg / renderer, and never on another feature. It is registered by name and
// created by the app at runtime, so adding an algorithm never touches the renderer core:
//
//   1. copy src/features/example/ to src/features/<name>/
//   2. implement Initialize() + AddPasses()
//   3. expose tunables through Parameters()
//   4. CMake picks it up automatically (air_add_feature)
//
// Features run by stage (FeatureStage), and within a stage in the order they were added to the
// FrameRunner, so a later feature can post-process an earlier one. Features never call each other:
// what one produces for another (denoiser guides, flags) goes through the frame's Blackboard
// under the names in namespace `slots`.
//
// Resolutions: the Scene stage renders at the render resolution into FrameContext::sceneColor;
// the output (and the Post stage) is at the output resolution. They are equal unless the
// FrameRunner was created with a render scale below 1, in which case an Upscale feature (dlss)
// reconstructs the output, or FrameRunner falls back to a bilinear upscale.

#include "air/core/Json.h"
#include "air/core/TaskSystem.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/Parameters.h"
#include "air/renderer/Scene.h"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace air::renderer
{

// Where a feature runs in the frame. Stages run in this order:
//   Scene    renders the scene at the render resolution into FrameContext::sceneColor
//   Upscale  reconstructs FrameContext::output from sceneColor and sets outputWritten (dlss)
//   Post     works on the output at the output resolution (tonemap, overlays)
// When sceneColor is smaller than the output and no Upscale feature wrote the output, FrameRunner
// adds a bilinear upscale between the Upscale and Post stages.
enum class FeatureStage : u8
{
    Scene,
    Upscale,
    Post,
};

struct FeatureInitContext
{
    rhi::Device& device;
    TaskSystem& tasks;
    std::filesystem::path shaderDir; // <shader root>/<feature name>
    u32 width = 0;                   // output resolution
    u32 height = 0;
    u32 renderWidth = 0; // Scene stage resolution; equals width/height without upscaling
    u32 renderHeight = 0;
};

// Well-known blackboard names. Producers document the format next to each name.
namespace slots
{
// Textures (all at the render resolution).
inline constexpr std::string_view kDiffuseAlbedo = "guide.diffuseAlbedo";     // RGBA16F, rgb
inline constexpr std::string_view kSpecularAlbedo = "guide.specularAlbedo";   // RGBA16F, rgb
inline constexpr std::string_view kNormalRoughness = "guide.normalRoughness"; // RGBA16F, world normal, a = roughness
inline constexpr std::string_view kViewDepth = "guide.viewDepth";             // R32F, metres along the view axis
inline constexpr std::string_view kMotionVectors = "guide.motionVectors";     // RG16F, previous - current, pixels
// RGBA16F, like sceneColor: the frame before transparent media (real-time volumes) were laid over it.
// Consumed by a denoiser of the whole frame, which also wants the guides to describe those media where
// they hide the surfaces.
inline constexpr std::string_view kColorBeforeTransparency = "guide.colorBeforeTransparency";
// Noisy signals for denoisers that work on the surfaces' light alone (RGBA16F, scene-linear like
// sceneColor): the light the first surface reflects through its diffuse base and through its specular
// lobes (not divided by the albedos), a = the length of its reflected ray (0: none). Its own emission,
// the sky, and media in front of it are in neither.
inline constexpr std::string_view kDiffuseSignal = "signal.diffuse";
inline constexpr std::string_view kSpecularSignal = "signal.specular";
// R16F: how much of the surface's light reaches the camera through media in front of it (real-time
// volumes); absent means 1. sceneColor = what lies in front + this * (emission + diffuse + specular).
inline constexpr std::string_view kSurfaceTransmittance = "signal.surfaceTransmittance";
// Values.
inline constexpr std::string_view kJitterX = "camera.jitterX"; // sub-pixel offset of the frame's samples, pixels
inline constexpr std::string_view kJitterY = "camera.jitterY";
// 1 when temporal history is meaningless (first frame, scene or render settings changed); camera
// motion alone does not reset it.
inline constexpr std::string_view kHistoryReset = "history.reset";
// 1 when sceneColor holds this frame's samples alone (at the jitter above), for temporal filters; 0
// when the producer averages frames itself (accumulation).
inline constexpr std::string_view kRealTime = "frame.realTime";
// 1 when the output is already display-referred (debug views): no tone curve.
inline constexpr std::string_view kOutputIsDisplay = "output.isDisplay";
} // namespace slots

// Per-frame named exchange between features. Lives for one RenderFrame().
class Blackboard
{
public:
    void SetTexture(std::string_view name, rdg::TextureRef texture) { mTextures[std::string(name)] = texture; }
    std::optional<rdg::TextureRef> FindTexture(std::string_view name) const
    {
        const auto it = mTextures.find(name);
        return it == mTextures.end() ? std::nullopt : std::optional(it->second);
    }
    void SetValue(std::string_view name, double value) { mValues[std::string(name)] = value; }
    double GetValue(std::string_view name, double fallback = 0.0) const
    {
        const auto it = mValues.find(name);
        return it == mValues.end() ? fallback : it->second;
    }

private:
    std::map<std::string, rdg::TextureRef, std::less<>> mTextures;
    std::map<std::string, double, std::less<>> mValues;
};

struct FrameContext
{
    rhi::Device& device;
    TaskSystem& tasks;
    rdg::RenderGraph& graph;
    const GpuScene& scene;
    const Camera& camera;
    Blackboard& blackboard;
    rdg::TextureRef output; // RGBA16F linear, UnorderedAccess | ShaderResource, output resolution
    u32 width = 0;          // output resolution
    u32 height = 0;
    u64 frameIndex = 0;
    // Scene stage target: RGBA16F at the render resolution, UnorderedAccess | ShaderResource. The
    // same texture as `output` when the render and output resolutions match.
    rdg::TextureRef sceneColor;
    u32 renderWidth = 0;
    u32 renderHeight = 0;
    // Set by an Upscale feature that writes `output` from `sceneColor` this frame.
    bool outputWritten = false;
    // Blackboard slots some active feature reads (RenderFeature::ConsumedSlots()). Producers skip
    // work nobody consumes, e.g. the path tracer's denoiser guides without a denoiser.
    const std::set<std::string, std::less<>>* consumedSlots = nullptr;
    bool Consumes(std::string_view slot) const { return consumedSlots && consumedSlots->contains(slot); }
};

class RenderFeature
{
public:
    virtual ~RenderFeature() = default;

    // Create pipelines and persistent resources. Return false (and log why) to refuse to run.
    virtual bool Initialize(const FeatureInitContext& context) = 0;
    // Add this frame's passes to context.graph. Called once per frame on the frame thread.
    virtual void AddPasses(FrameContext& context) = 0;
    // Feature specific state for reports (e.g. accumulated samples). Optional.
    virtual json::Value GetStatus() const { return {}; }
    // Fixed for the feature's lifetime. Post is right for anything that works on the output.
    virtual FeatureStage GetStage() const { return FeatureStage::Post; }
    // Blackboard slots this feature reads, so their producers know someone needs them. Asked every
    // frame: a feature that a parameter switched off returns none.
    virtual std::vector<std::string_view> ConsumedSlots() const { return {}; }

    ParameterSet& Parameters() { return mParameters; }
    const ParameterSet& Parameters() const { return mParameters; }

protected:
    ParameterSet mParameters;
};

struct FeatureInfo
{
    std::string name;        // identifier used on the command line, e.g. "pathtracer"
    std::string description; // one line
    std::function<std::unique_ptr<RenderFeature>()> create;
};

class FeatureRegistry
{
public:
    void Register(FeatureInfo info);
    std::unique_ptr<RenderFeature> Create(std::string_view name) const;
    const FeatureInfo* Find(std::string_view name) const;
    const std::vector<FeatureInfo>& List() const { return mFeatures; }

private:
    std::vector<FeatureInfo> mFeatures;
};

// Registers every feature compiled into the binary. Generated by CMake from air_add_feature().
void RegisterBuiltinFeatures(FeatureRegistry& registry);

} // namespace air::renderer
