#pragma once

// Progressive GPU path tracer (offline / film quality, and interactive preview).
//
// Each frame AddPasses() appends these render graph passes:
//   PathTracer.Trace    DispatchRays: N samples per pixel into a persistent RGBA32F accumulation
//                       buffer, plus first-hit guide textures for denoisers
//   PathTracer.ReSTIR*  Compute, with restir.di / restir.gi: the direct / indirect light at the
//                       first surface by ReSTIR DI / GI (src/ReSTIR.h), added to the accumulation
//   PathTracer.Resolve  Compute: average * exposure (or a debug view) into the frame output
// The estimator (shaders/Integrator.hlsli: NEE + MIS over emissive triangles, energy-conserving
// GGX) is shared with the CPU ReferenceRenderer. The output is scene-linear and exposed
// (radiance * ExposureScale(EV100)); the `tonemap` feature turns it into a display image.

#include "air/rdg/RenderGraph.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"
#include "air/rhi/RHIDevice.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace air::pt
{

// ReSTIR (docs/architecture.md §8.7, §8.8): the light at each pixel's first surface, resampled
// across pixels and frames instead of estimated by each path alone.
// When ReSTIR DI / GI run. Reuse pays off in single frames (real-time, under a denoiser); when
// frames accumulate, plain paths converge about as fast for the same time.
inline constexpr u32 kReSTIROff = 0;
inline constexpr u32 kReSTIRRealTime = 1; // only with accumulate=false
inline constexpr u32 kReSTIRAlways = 2;   // also when accumulating (e.g. to check it is unbiased)

struct ReSTIRSettings
{
    u32 di = kReSTIRRealTime;    // direct light (shaders/ReSTIRDI.hlsli) instead of one NEE sample
    u32 gi = kReSTIRRealTime;    // indirect light, all paths of two or more bounces (shaders/ReSTIRGI.hlsli)
    u32 lightCandidates = 8;     // initial candidates: light points chosen by power
    u32 bsdfCandidates = 1;      // initial candidates: BSDF rays that may reach an emitter
    u32 spatialSamples = 2;      // neighbours reused per pixel
    u32 disocclusionSamples = 8; // neighbours reused where a pixel has no temporal history
    float spatialRadius = 30.0f; // pixels
    u32 maxConfidence = 20;      // cap of the temporal history, in frames' worth of candidates
    // Trace the visibility the MIS weights need (one shadow ray per reused reservoir). Off: fewer
    // rays, but penumbrae come out darker (the classic biased visibility reuse).
    bool unbiased = true;

    bool operator==(const ReSTIRSettings&) const = default;
};

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
    // NEE at the first lightTreeBounces path vertices chooses lights with the light tree (power,
    // distance and orientation from the vertex), deeper ones by power alone (O(1)). 0: by power only.
    // At every vertex the tree halves the noise of many_lights per sample but costs ~3x the trace
    // time (a walk of ~log2(lights) dependent loads per sample): off by default, worth it for
    // offline references of many-light scenes (docs/architecture.md §8.6).
    u32 lightTreeBounces = 0;
    ReSTIRSettings restir;
    bool furnace = false;     // white materials, emitters off, auto exposure on the environment
    u32 debugView = 0;        // gpu::kDebug*
    float debugScale = 10.0f; // metres for the depth view, intersections for the heatmap
    // Length of the Halton(2, 3) jitter sequence used with accumulate=false. Fixed after Initialize().
    u32 jitterPhases = 32;
    // DXR 1.2 shader execution reordering when the device actually reorders. Off by default: with
    // this kernel's cheap shading the reorder costs more than it saves (docs/architecture.md §8.2).
    bool shaderExecutionReordering = false;
};

class ReSTIR;

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
    // True when this frame's trace uses the shader execution reordering pipeline (not with ReSTIR GI,
    // which has a kernel of its own).
    bool UsesShaderExecutionReordering() const
    {
        return mSettings.shaderExecutionReordering && mTracePipelineSer.IsValid() && !mUsedReSTIRGI;
    }
    bool IsConverged() const { return mSettings.accumulate && mAccumulatedSamples >= mSettings.targetSamples; }
    // Whether the last traced frame's direct / indirect light came from ReSTIR DI / GI.
    bool UsesReSTIRDI() const { return mUsedReSTIRDI; }
    bool UsesReSTIRGI() const { return mUsedReSTIRGI; }
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
    rhi::PipelineHandle mTracePipelineSer; // PathTracerSer.lib (SM 6.9), when the device reorders
    rhi::PipelineHandle mTracePipelineGi;  // PathTracerGi.lib: records ReSTIR GI's candidates
    rhi::PipelineHandle mResolvePipeline;
    rhi::TextureHandle mAccumulation;
    rhi::Access mAccumulationAccess = rhi::Access::None;
    rhi::BufferHandle mBsdfLut;
    rhi::BufferHandle mFrameRing; // StructuredBuffer<FrameData>, framesInFlight + 1 elements
    u32 mFrameRingSize = 0;
    std::unique_ptr<ReSTIR> mReSTIR;
    bool mUsedReSTIRDI = false;
    bool mUsedReSTIRGI = false;

    u32 mAccumulatedSamples = 0;
    u32 mFrameIndex = 0;
    bool mResetPending = true;
};

} // namespace air::pt
