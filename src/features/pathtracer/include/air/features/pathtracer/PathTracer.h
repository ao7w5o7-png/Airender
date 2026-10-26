#pragma once

// Progressive GPU path tracer (offline / film quality, and interactive preview).
//
// Each frame AddPasses() appends these render graph passes:
//   PathTracer.Trace    DispatchRays: N samples per pixel into a persistent RGBA32F accumulation
//                       buffer, plus first-hit guide textures for denoisers
//   PathTracer.ReSTIR*  Compute, with restir.di / restir.gi: the direct / indirect light at the
//                       first surface by ReSTIR DI / GI (src/ReSTIR.h), added to the accumulation
//   NRC.*               Compute, with nrc: the neural radiance cache's training (src/RadianceCache.h)
//   VolumeMarch.*       Compute, with volumeMarch: the volumes ray marched over the surfaces
//                       (src/VolumeMarcher.h)
//   PathTracer.Resolve  Compute: average * exposure (or a debug view) into the frame output
// The estimator (shaders/Integrator.hlsli: NEE + MIS over emissive triangles, energy-conserving
// GGX) is shared with the CPU ReferenceRenderer. The output is scene-linear and exposed
// (radiance * ExposureScale(EV100)); the `tonemap` feature turns it into a display image.

#include "air/rdg/RenderGraph.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"
#include "air/renderer/ShaderLoader.h"
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

// Neural radiance cache (docs/architecture.md §8.11; Mueller et al. 2021): paths end where their
// footprint has spread enough, usually at the first diffuse bounce, and take the light reflected
// there from a small network trained every frame on a few longer paths. Far less noise and fewer
// rays per pixel than following the path, at some bias; it replaces ReSTIR GI while on.
inline constexpr u32 kNrcOff = 0;
inline constexpr u32 kNrcRealTime = 1; // only with accumulate=false
inline constexpr u32 kNrcAlways = 2;   // also when accumulating (e.g. to measure its bias)

struct RadianceCacheSettings
{
    u32 mode = kNrcOff;
    // A path ends where its footprint's spread exceeds this times the primary vertex's (paper: 0.01).
    float spread = 0.01f;
    u32 trainingPaths = 1u << 15; // per frame, one per tile of pixels
    u32 steps = 4;                // optimizer steps per frame, over the training paths
    float learningRate = 1e-2f;
    u32 unbiasedPeriod = 16; // 1 in this many training paths is never ended by the cache
    u32 tableLog2 = 16;      // hash grid entries per level, log2 (a change starts a new network)
    // On the GPU's matrix units (Shader Model 6.10 linear algebra) where the device has them; off:
    // the float32 kernels everywhere.
    bool linearAlgebra = true;

    bool operator==(const RadianceCacheSettings&) const = default;
};

// Real-time volumes (docs/architecture.md §8.13): the scene's volumes ray marched over the surfaces
// and lit through moment shadow maps, instead of tracked by every path. Light is scattered once on its
// way to the camera and the shadows are approximations: biased, but free of noise and a fraction of
// the cost of tracking.
inline constexpr u32 kVolumeMarchOff = 0;
inline constexpr u32 kVolumeMarchRealTime = 1; // only with accumulate=false
inline constexpr u32 kVolumeMarchAlways = 2;   // also when accumulating (e.g. to measure its bias)
inline constexpr u32 kMaxVolumeLights = 8;     // light groups with moment shadow maps (VolumeMarchShared.h)
inline constexpr u32 kMaxLightOrigins = 4;     // maps of one of them (VolumeMarchShared.h)

struct VolumeMarchSettings
{
    u32 mode = kVolumeMarchRealTime;
    float step = 1.0f;             // the march's step, in voxels of `level`
    u32 level = 0;                 // the level of detail of the bricks it reads (0 the finest, 2 the coarsest)
    u32 shadowSize = 256;          // texels per side of every moment shadow map
    u32 shadowLevel = 1;           // the level their marches read
    float shadowStep = 1.0f;       // their step, in voxels of that level
    u32 lights = kMaxVolumeLights; // light groups that light the volumes (those that matter most to them)
    // Shadow maps of a large emitter close to the volumes, from points spread over it (1: its centre).
    u32 lightOrigins = kMaxLightOrigins;
    bool air = true; // the aerial perspective dims the volumes at the depth their light comes from
    // Adaptive sampling groups: a wave of pixels whose transmittance is below shareBelow2 lights each
    // step once per 2 x 2 pixels, below shareBelow4 once per 4 x 4 (VolumeMarch.hlsl); so does one
    // whose steps lie where a voxel covers shareFootprint2 (shareFootprint4) pixels or more.
    bool adaptive = true;
    float shareBelow2 = 0.3f;
    float shareBelow4 = 0.1f;
    float shareFootprint2 = 2.0f;
    float shareFootprint4 = 4.0f;
    // Render the shadow maps every frame (to measure them) instead of when what they show changed.
    bool everyFrame = false;
    // In real time without a denoiser of the whole frame: filter the volumes' layer over frames
    // (VolumeTemporal.hlsl), this frame's weight temporalBlend.
    bool temporal = true;
    float temporalBlend = 0.1f;

    bool operator==(const VolumeMarchSettings&) const = default;
};

// Clear coats (docs/architecture.md §8.17): how coated materials evaluate the light their coat lets in
// and out again.
inline constexpr u32 kCoatModelNeural = 0;      // an MLP trained for the scene's coated materials
inline constexpr u32 kCoatModelRandomWalk = 1;  // the reference: a random walk per evaluation (noisy)
inline constexpr u32 kCoatModelApproximate = 2; // one bounce through a smooth coat, analytic

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
    // Set by the feature from what other features consume (not parameters): the first surface's light
    // split by lobe (slots::kDiffuseSignal ...), and, under real-time volumes, guides that describe the
    // volumes where they hide the surfaces with the frame without them (slots::kColorBeforeTransparency).
    bool signals = false;
    bool transparencyGuides = false;
    bool nextEventEstimation = true;
    // NEE at the first lightTreeBounces path vertices chooses lights with the light tree (power,
    // distance and orientation from the vertex), deeper ones by power alone (O(1)). 0: by power only.
    // At every vertex the tree halves the noise of many_lights per sample but costs ~3x the trace
    // time (a walk of ~log2(lights) dependent loads per sample): off by default, worth it for
    // offline references of many-light scenes (docs/architecture.md §8.6).
    u32 lightTreeBounces = 0;
    // Scattering events a path's walk through the clouds may take before it is given up (its light
    // lost). Thick clouds scatter light hundreds of times before it leaves them.
    u32 cloudEvents = 256;
    // Scattering events a path may take in the scene's volume, on top of its maxBounces surfaces,
    // before its light is given up. A bright medium (albedo near 1) needs hundreds.
    u32 volumeEvents = 256;
    ReSTIRSettings restir;
    RadianceCacheSettings nrc;
    VolumeMarchSettings volumeMarch;
    bool furnace = false;     // white materials, emitters off, auto exposure on the environment
    u32 debugView = 0;        // gpu::kDebug*
    float debugScale = 10.0f; // metres for the depth view, intersections for the heatmap
    // Length of the Halton(2, 3) jitter sequence used with accumulate=false. Fixed after Initialize().
    u32 jitterPhases = 32;
    // DXR 1.2 shader execution reordering when the device actually reorders. Off by default: with
    // this kernel's cheap shading the reorder costs more than it saves (docs/architecture.md §8.2).
    bool shaderExecutionReordering = false;
    u32 coatModel = kCoatModelNeural;
};

class ReSTIR;
class RadianceCache;
class VolumeMarcher;

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
    // Where the neural coat trains (TrainNeuralCoat) when a scene's coated materials change; null: on
    // the frame thread alone.
    void SetTaskSystem(TaskSystem* tasks) { mTasks = tasks; }
    // The neural coat of the scene's coated materials: its training loss (0: none), and whether it came
    // from the cache.
    float GetNeuralCoatLoss() const { return mNeuralCoatLoss; }

    // Adds this frame's passes. `output` must allow UnorderedAccess and be RGBA16F/RGBA32F.
    // Restarts accumulation by itself when the camera, scene or an image-affecting setting changed.
    void AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const renderer::Camera& camera,
                   renderer::Blackboard& blackboard, rdg::TextureRef output);

    u32 GetAccumulatedSamples() const { return mAccumulatedSamples; }
    // True when this frame's trace uses the shader execution reordering pipeline (not with ReSTIR GI,
    // which has a kernel of its own).
    bool UsesShaderExecutionReordering() const
    {
        return mSettings.shaderExecutionReordering && mTracePipelineSer.IsValid() && !mUsedReSTIRGI &&
               !mUsedNeuralKernels;
    }
    bool IsConverged() const { return mSettings.accumulate && mAccumulatedSamples >= mSettings.targetSamples; }
    // Whether the last traced frame's direct / indirect light came from ReSTIR DI / GI.
    bool UsesReSTIRDI() const { return mUsedReSTIRDI; }
    bool UsesReSTIRGI() const { return mUsedReSTIRGI; }
    // Whether the last traced frame ran the kernels with the neural parts (AIR_PT_NEURAL: neural texture
    // sets decoded on sample, coated materials' models).
    bool UsesNeuralKernels() const { return mUsedNeuralKernels; }
    // Whether the last traced frame's paths ended in the neural radiance cache, and whether it ran
    // on the matrix units (Shader Model 6.10 linear algebra).
    bool UsesRadianceCache() const { return mUsedNrc; }
    // Whether the last traced frame's volumes were ray marched (not tracked), and with how many light
    // groups besides the sun.
    bool UsesVolumeMarch() const { return mUsedVolumeMarch; }
    u32 GetVolumeMarchLights() const;
    bool RadianceCacheUsesLinearAlgebra() const;
    u32 GetRadianceCacheSteps() const;
    const PathTracerSettings& GetSettings() const { return mSettings; }
    // Settings other than width/height may change between frames.
    PathTracerSettings& MutableSettings() { return mSettings; }

private:
    // Trains (or finds) the neural coat when the scene's coated materials changed.
    void UpdateNeuralCoat(const renderer::GpuScene& scene);

    rhi::Device& mDevice;
    PathTracerSettings mSettings;
    PathTracerSettings mLastSettings;
    std::optional<renderer::Camera> mLastCamera;
    u64 mLastSceneVersion = ~0ull;

    rhi::PipelineHandle mTracePipeline;
    rhi::PipelineHandle mTracePipelineSer; // PathTracerSer.lib (SM 6.9), when the device reorders
    rhi::PipelineHandle mTracePipelineGi;  // PathTracerGi.lib: records ReSTIR GI's candidates
    // The kernel with the scene's volume (AIR_PT_VOLUME), created the first time a scene has one.
    rhi::RayTracingPipelineDesc mTraceVolumeDesc;
    rhi::PipelineHandle mTracePipelineVolume;
    // The kernel under real-time volumes (AIR_PT_VOLUME_SHADOWS), created the first time they run.
    rhi::RayTracingPipelineDesc mTraceVolumeShadowsDesc;
    rhi::PipelineHandle mTracePipelineVolumeShadows;
    // The kernels with the neural parts (AIR_PT_NEURAL: neural texture sets decoded on sample, the clear
    // coats' models), PathTracerNeural.lib and PathTracerGiNeural.lib, created the first time a scene
    // needs them.
    rhi::RayTracingPipelineDesc mTraceNeuralDesc;
    rhi::RayTracingPipelineDesc mTraceGiNeuralDesc;
    rhi::PipelineHandle mTracePipelineNeural;
    rhi::PipelineHandle mTracePipelineGiNeural;
    // PathTracerNrc.cs (paths end in the cache, linear algebra) and PathTracerNrcPortable.cs (float32),
    // created the first time the cache runs.
    renderer::LazyComputePipeline mTraceNrc;
    renderer::LazyComputePipeline mTraceNrcPortable;
    rhi::PipelineHandle mResolvePipeline;
    // The neural coat: its parameters (ByteAddressBuffer) for the coated materials it was trained on.
    TaskSystem* mTasks = nullptr;
    rhi::BufferHandle mNeuralCoat;
    u64 mNeuralCoatKey = 0;
    float mNeuralCoatLoss = 0.0f;
    static constexpr u32 kNeuralCoatSettleFrames = 30; // unchanged frames before a new network trains
    bool mNeuralCoatPending = true;
    u32 mNeuralCoatStableFrames = 0;
    rhi::TextureHandle mAccumulation;
    rhi::Access mAccumulationAccess = rhi::Access::None;
    rhi::BufferHandle mBsdfLut;
    rhi::BufferHandle mFrameRing; // StructuredBuffer<FrameData>, framesInFlight + 1 elements
    u32 mFrameRingSize = 0;
    std::unique_ptr<ReSTIR> mReSTIR;
    std::unique_ptr<RadianceCache> mRadianceCache;
    std::unique_ptr<VolumeMarcher> mVolumeMarcher;
    bool mUsedVolumeMarch = false;
    bool mUsedReSTIRDI = false;
    bool mUsedReSTIRGI = false;
    bool mUsedNeuralKernels = false;
    bool mUsedNrc = false;

    u32 mAccumulatedSamples = 0;
    u32 mFrameIndex = 0;
    bool mResetPending = true;
};

} // namespace air::pt
