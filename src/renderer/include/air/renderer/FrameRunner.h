#pragma once

// FrameRunner: owns the per-frame loop (graph creation, feature passes, execute, reporting).
// Apps, tests and the AI harness all drive rendering through this one class, so they exercise the
// exact same code path.

#include "air/core/Json.h"
#include "air/core/Profiler.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/RenderFeature.h"

#include <memory>
#include <string>
#include <vector>

namespace air::renderer {

struct FrameRunnerDesc {
    u32 width = 1280;
    u32 height = 720;
    rhi::Format outputFormat = rhi::Format::RGBA16_FLOAT;
    std::filesystem::path shaderRoot = "shaders"; // features load from shaderRoot/<feature name>
    rdg::RenderGraph::Config graphConfig;
    bool captureCommands = false; // keep each frame's command stream in FrameResult
    bool captureGraph = false;    // keep each frame's graph dump in FrameResult
    // GPU timestamps per pass, CPU recording time per pass and a memory snapshot in every
    // FrameResult (and in its JSON).
    bool profile = false;
};

struct FrameResult {
    u64 frameIndex = 0;
    rdg::RenderGraph::Stats graphStats;
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

    json::Value ToJson() const;
};

class FrameRunner {
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

    FrameResult RenderFrame(const GpuScene& scene, const Camera& camera);

    rhi::TextureHandle GetOutput() const { return mOutput; }
    const FrameRunnerDesc& GetDesc() const { return mDesc; }
    // Features with their parameters and status.
    json::Value DescribeFeatures() const;

private:
    struct Entry {
        std::string name;
        std::unique_ptr<RenderFeature> feature;
    };

    rhi::Device& mDevice;
    TaskSystem& mTasks;
    FrameRunnerDesc mDesc;
    std::vector<Entry> mFeatures;
    rhi::TextureHandle mOutput;
    rhi::Access mOutputAccess = rhi::Access::None;
    u64 mFrameIndex = 0;
};

} // namespace air::renderer
