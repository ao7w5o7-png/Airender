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
// Features run in the order they were added to the FrameRunner and all write to the same output
// texture, so a later feature can post-process an earlier one. Features never call each other:
// what one produces for another (denoiser guides, flags) goes through the frame's Blackboard
// under the names in namespace `slots`.

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
#include <string>
#include <vector>

namespace air::renderer {

struct FeatureInitContext {
    rhi::Device& device;
    TaskSystem& tasks;
    std::filesystem::path shaderDir; // <shader root>/<feature name>
    u32 width = 0;
    u32 height = 0;
};

// Well-known blackboard names. Producers document the format next to each name.
namespace slots {
// Textures (all render resolution).
inline constexpr std::string_view kDiffuseAlbedo = "guide.diffuseAlbedo";     // RGBA16F, rgb
inline constexpr std::string_view kSpecularAlbedo = "guide.specularAlbedo";   // RGBA16F, rgb
inline constexpr std::string_view kNormalRoughness = "guide.normalRoughness"; // RGBA16F, world normal, a = roughness
inline constexpr std::string_view kViewDepth = "guide.viewDepth";             // R32F, linear distance along -forward
inline constexpr std::string_view kMotionVectors = "guide.motionVectors";     // RG16F, pixels, current -> previous
// Values.
inline constexpr std::string_view kJitterX = "camera.jitterX"; // sub-pixel offset of the frame's samples, pixels
inline constexpr std::string_view kJitterY = "camera.jitterY";
inline constexpr std::string_view kCameraReset = "camera.reset"; // 1 when history must be discarded
inline constexpr std::string_view kOutputIsDisplay =
    "output.isDisplay"; // 1: output is already display-referred (debug views)
} // namespace slots

// Per-frame named exchange between features. Lives for one RenderFrame().
class Blackboard {
public:
    void SetTexture(std::string_view name, rdg::TextureRef texture) { mTextures[std::string(name)] = texture; }
    std::optional<rdg::TextureRef> FindTexture(std::string_view name) const {
        const auto it = mTextures.find(name);
        return it == mTextures.end() ? std::nullopt : std::optional(it->second);
    }
    void SetValue(std::string_view name, double value) { mValues[std::string(name)] = value; }
    double GetValue(std::string_view name, double fallback = 0.0) const {
        const auto it = mValues.find(name);
        return it == mValues.end() ? fallback : it->second;
    }

private:
    std::map<std::string, rdg::TextureRef, std::less<>> mTextures;
    std::map<std::string, double, std::less<>> mValues;
};

struct FrameContext {
    rhi::Device& device;
    TaskSystem& tasks;
    rdg::RenderGraph& graph;
    const GpuScene& scene;
    const Camera& camera;
    Blackboard& blackboard;
    rdg::TextureRef output; // RGBA16F linear, UnorderedAccess | ShaderResource
    u32 width = 0;
    u32 height = 0;
    u64 frameIndex = 0;
};

class RenderFeature {
public:
    virtual ~RenderFeature() = default;

    // Create pipelines and persistent resources. Return false (and log why) to refuse to run.
    virtual bool Initialize(const FeatureInitContext& context) = 0;
    // Add this frame's passes to context.graph. Called once per frame on the frame thread.
    virtual void AddPasses(FrameContext& context) = 0;
    // Feature specific state for reports (e.g. accumulated samples). Optional.
    virtual json::Value GetStatus() const { return {}; }

    ParameterSet& Parameters() { return mParameters; }
    const ParameterSet& Parameters() const { return mParameters; }

protected:
    ParameterSet mParameters;
};

struct FeatureInfo {
    std::string name;        // identifier used on the command line, e.g. "pathtracer"
    std::string description; // one line
    std::function<std::unique_ptr<RenderFeature>()> create;
};

class FeatureRegistry {
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
