#pragma once

// Progressive GPU path tracer (offline / film quality, and interactive preview).
//
// Each frame AddPasses() appends two render graph passes:
//   PathTracer.Trace    DispatchRays: N samples per pixel into a persistent RGBA32F accumulation
//                       buffer, plus first-hit guide textures for denoisers
//   PathTracer.Resolve  Compute: average * exposure (or a debug view) into the frame output
// The estimator (shaders/Integrator.hlsli: NEE + MIS over emissive triangles, energy-conserving
// GGX) is shared with the CPU ReferenceRenderer. The output is scene-linear and exposed
// (radiance * ExposureScale(EV100)); the `tonemap` feature turns it into a display image.

#include "air/rdg/RenderGraph.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"
#include "air/rhi/RHIDevice.h"

#include <filesystem>
#include <optional>

namespace air::pt
{

struct PathTracerSettings
{
    u32 width = 1920;
    u32 height = 1080;
    u32 samplesPerPixelPerFrame = 4; // raise for offline throughput, lower for interactive preview
    u32 maxBounces = 8;
    u32 targetSamples = 1024; // IsConverged() once reached; tracing stops there
    // false: every frame shows only its own samples (input for a real-time denoiser such as DLSS RR,
    // which needs the frame's jitter from the blackboard).
    bool accumulate = true;
    bool guides = true; // write the denoiser guide textures (blackboard slots::kDiffuseAlbedo ...)
    bool nextEventEstimation = true;
    bool furnace = false;     // white materials, emitters off, auto exposure on the environment
    u32 debugView = 0;        // gpu::kDebug*
    float debugScale = 10.0f; // metres for the depth view, intersections for the heatmap
};

class PathTracer
{
public:
    PathTracer(rhi::Device& device, const PathTracerSettings& settings);
    ~PathTracer();

    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

    // Loads compiled shaders from shaderDir (PathTracer.lib.dxil|spv, Resolve.cs.dxil|spv) and
    // uploads the BSDF tables. Settings that size resources (width/height) are fixed after this.
    bool Initialize(const std::filesystem::path& shaderDir);

    void ResetAccumulation();

    // Adds this frame's passes. `output` must allow UnorderedAccess and be RGBA16F/RGBA32F.
    // Restarts accumulation by itself when the camera, scene or an image-affecting setting changed.
    void AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const renderer::Camera& camera,
                   renderer::Blackboard& blackboard, rdg::TextureRef output);

    u32 GetAccumulatedSamples() const { return mAccumulatedSamples; }
    bool IsConverged() const { return mSettings.accumulate && mAccumulatedSamples >= mSettings.targetSamples; }
    const PathTracerSettings& GetSettings() const { return mSettings; }
    // Settings other than width/height may change between frames.
    PathTracerSettings& MutableSettings() { return mSettings; }

private:
    rhi::Device& mDevice;
    PathTracerSettings mSettings;
    PathTracerSettings mLastSettings;
    std::optional<renderer::Camera> mLastCamera;
    u64 mLastSceneVersion = ~0ull;

    rhi::PipelineHandle mTracePipeline;
    rhi::PipelineHandle mResolvePipeline;
    rhi::TextureHandle mAccumulation;
    rhi::Access mAccumulationAccess = rhi::Access::None;
    rhi::BufferHandle mBsdfLut;
    rhi::BufferHandle mFrameRing; // StructuredBuffer<FrameData>, framesInFlight + 1 elements
    u32 mFrameRingSize = 0;

    u32 mAccumulatedSamples = 0;
    u32 mFrameIndex = 0;
    bool mResetPending = true;
};

} // namespace air::pt
