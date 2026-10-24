#pragma once

// FrameRunner: owns the per-frame loop (graph creation, feature passes, execute, reporting).
// Apps, tests and the AI harness all drive rendering through this one class, so they exercise the
// exact same code path.

#include "air/core/Json.h"
#include "air/core/Profiler.h"
#include "air/rdg/RenderGraph.h"
#include "air/rdg/TransientResourcePool.h"
#include "air/renderer/RenderFeature.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace air::renderer
{

struct FrameRunnerDesc
{
    u32 width = 1280; // output resolution
    u32 height = 720;
    // Scene stage resolution as a fraction of the output per axis (1 = native). Below 1 an Upscale
    // feature (dlss) reconstructs the output; without one FrameRunner upscales bilinearly.
    float renderScale = 1.0f;
    rhi::Format outputFormat = rhi::Format::RGBA16_FLOAT;
    std::filesystem::path shaderRoot = "shaders"; // features load from shaderRoot/<feature name>
    rdg::RenderGraph::Config graphConfig;
    // Keep the transient heap and placed resources from frame to frame (rdg::TransientResourcePool);
    // off: every frame creates and releases them, as a render graph without a pool does.
    bool poolTransients = true;
    bool captureCommands = false; // keep each frame's command stream in FrameResult
    bool captureGraph = false;    // keep each frame's graph dump in FrameResult
    // GPU timestamps per pass, CPU recording time per pass and a memory snapshot in every
    // FrameResult (and in its JSON).
    bool profile = false;
};

// Everything a profile report needs from one frame (ProfileReport.h), kept when
// FrameRunner::RequestCapture() asked for it. GPU timings resolve a few frames later: the frame's
// are the GpuFrameTimings whose frameIndex equals deviceFrame.
struct FrameCapture
{
    u64 frameIndex = 0;  // FrameRunner frame
    u64 deviceFrame = 0; // rhi::Device::GetFrameIndex() while it was recorded
    u32 width = 0;       // output resolution
    u32 height = 0;
    u32 renderWidth = 0; // Scene stage resolution
    u32 renderHeight = 0;
    json::Value graph;                        // rdg::RenderGraph::DumpJson()
    std::vector<std::string> commands;        // the command stream, "[Graphics] Dispatch 8x4x1"
    std::vector<rhi::ResourceInfo> resources; // live resources after the frame was submitted
    rhi::MemoryStats memory;                  // idem
};

struct FrameResult
{
    u64 frameIndex = 0;
    rdg::RenderGraph::Stats graphStats;
    rdg::TransientResourcePool::Stats transientPool; // after this frame; zero without a pool
    double cpuMilliseconds = 0.0;
    std::string graphText;             // when captureGraph
    std::string graphviz;              // when captureGraph
    std::vector<std::string> commands; // when captureCommands
    std::vector<std::string> validationErrors;
    // When FrameRunnerDesc::profile is set:
    std::vector<rdg::RenderGraph::PassTiming> passCpu; // this frame
    rhi::GpuFrameTimings gpu;                          // latest frame the GPU finished (a few frames back)
    rhi::MemoryStats memory;                           // after this frame
    ProcessMemory processMemory;
    std::shared_ptr<const FrameCapture> capture; // when FrameRunner::RequestCapture() asked for this frame

    json::Value ToJson() const;
};

class FrameRunner
{
public:
    FrameRunner(rhi::Device& device, TaskSystem& tasks, FrameRunnerDesc desc);
    ~FrameRunner();

    FrameRunner(const FrameRunner&) = delete;
    FrameRunner& operator=(const FrameRunner&) = delete;

    // Initializes the feature and appends it to the frame. Returns false if it refused to initialize.
    bool AddFeature(const std::string& name, std::unique_ptr<RenderFeature> feature);
    RenderFeature* FindFeature(std::string_view name) const;
    // "feature.parameter" = value, e.g. SetParameter("pathtracer.maxBounces", "4", &error).
    bool SetParameter(std::string_view qualifiedName, std::string_view value, std::string* error = nullptr);

    // `extraPasses` runs after every feature, in the same graph (e.g. an editor drawing its UI
    // into the swapchain while reading the output).
    FrameResult RenderFrame(const GpuScene& scene, const Camera& camera,
                            const std::function<void(FrameContext&)>& extraPasses = {});
    // The next RenderFrame() fills FrameResult::capture, for a profile report.
    void RequestCapture() { mCaptureRequested = true; }

    rhi::TextureHandle GetOutput() const { return mOutput; }
    const FrameRunnerDesc& GetDesc() const { return mDesc; }
    // Every frame builds a new graph, so this takes effect on the next RenderFrame().
    void SetGraphConfig(const rdg::RenderGraph::Config& config) { mDesc.graphConfig = config; }
    // Scene stage resolution: the output resolution scaled by FrameRunnerDesc::renderScale.
    u32 GetRenderWidth() const { return mRenderWidth; }
    u32 GetRenderHeight() const { return mRenderHeight; }
    // Features with their parameters and status.
    json::Value DescribeFeatures() const;

private:
    void AddFallbackUpscale(FrameContext& context);

    struct Entry
    {
        std::string name;
        std::unique_ptr<RenderFeature> feature;
    };

    rhi::Device& mDevice;
    TaskSystem& mTasks;
    FrameRunnerDesc mDesc;
    std::vector<Entry> mFeatures;
    std::unique_ptr<rdg::TransientResourcePool> mTransientPool; // when FrameRunnerDesc::poolTransients
    rhi::TextureHandle mOutput;
    rhi::Access mOutputAccess = rhi::Access::None;
    u64 mFrameIndex = 0;
    bool mCaptureRequested = false;
    u32 mRenderWidth = 0;
    u32 mRenderHeight = 0;
    rhi::PipelineHandle mUpscalePipeline;              // fallback upscale, only when rendering below the output size
    std::set<std::string, std::less<>> mConsumedSlots; // union of the features' ConsumedSlots()
};

} // namespace air::renderer
