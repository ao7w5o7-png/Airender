#pragma once

// Progressive GPU path tracer for offline / film quality frames.
//
// Each frame AddPasses() appends two render graph passes:
//   PathTracer.Trace    DispatchRays: N samples per pixel, accumulated into a persistent RGBA32F buffer
//   PathTracer.Resolve  Compute: average + exposure + tonemap into the caller's output texture
// The accumulation buffer lives across frames, so an offline render is just "run frames until
// GetAccumulatedSamples() >= targetSamples". Changing the camera or scene calls ResetAccumulation().

#include "air/rdg/RenderGraph.h"
#include "air/renderer/Scene.h"
#include "air/rhi/RHIDevice.h"

#include <filesystem>

namespace air::pt {

struct Camera {
    float position[3] = {0, 1, 4};
    float forward[3] = {0, 0, -1};
    float up[3] = {0, 1, 0};
    float verticalFovRadians = 0.785398f;
};

struct PathTracerSettings {
    u32 width = 1920;
    u32 height = 1080;
    u32 samplesPerPixelPerFrame = 4; // raise for offline throughput, lower for interactive preview
    u32 maxBounces = 8;
    u32 targetSamples = 1024; // IsConverged() once reached
    float exposure = 1.0f;
};

class PathTracer {
public:
    PathTracer(rhi::Device& device, const PathTracerSettings& settings);
    ~PathTracer();

    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

    // Loads compiled shaders from shaderDir (PathTracer.lib.dxil|spv, Resolve.cs.dxil|spv).
    // The Null backend accepts missing files so the pipeline can run headless.
    // Settings that size resources (width/height) are fixed after this call.
    bool Initialize(const std::filesystem::path& shaderDir);

    void SetCamera(const Camera& camera);
    void ResetAccumulation();

    // Adds this frame's passes. `output` must allow UnorderedAccess and be RGBA8/RGBA16F/RGBA32F.
    void AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, rdg::TextureRef output);

    u32 GetAccumulatedSamples() const { return mAccumulatedSamples; }
    bool IsConverged() const { return mAccumulatedSamples >= mSettings.targetSamples; }
    const PathTracerSettings& GetSettings() const { return mSettings; }
    // Settings that do not size resources (spp, bounces, exposure, target) may change between frames.
    PathTracerSettings& MutableSettings() { return mSettings; }

private:
    rhi::Device& mDevice;
    PathTracerSettings mSettings;
    Camera mCamera;

    rhi::PipelineHandle mTracePipeline;
    rhi::PipelineHandle mResolvePipeline;
    rhi::TextureHandle mAccumulation;
    rhi::Access mAccumulationAccess = rhi::Access::None;

    u32 mAccumulatedSamples = 0;
    u32 mFrameIndex = 0;
    bool mResetPending = true;
};

} // namespace air::pt
