#pragma once

// The real-time volumes (docs/architecture.md §8.13): instead of being tracked by every path, the
// scene's volumes are ray marched over the path traced surfaces, lit by the sun and a few light groups
// through moment shadow maps rendered once per light, and by the light around them (probes). The
// design follows Mueller's description of a production volume renderer (SIGGRAPH 2026).
//
// Each frame, around PathTracer.Trace (whose kernel variant PathTracerVolumeShadows.hlsl lights the
// surfaces through the same maps and writes the frame's samples and their first hit's distance
// instead of accumulating them):
//   VolumeMarch.Shadows  Compute, async: the moment shadow maps, again only when what they depend on
//                        (the volumes, the lights, the sun, the settings) changed
//   VolumeMarch.Probes   Compute, async (it overlaps the trace): paths from 64 probes per volume
//   VolumeMarch.March    Compute: every pixel's camera ray through the volumes, over the trace's
//                        samples, into the accumulation

#include "air/features/pathtracer/PathTracer.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/Scene.h"
#include "air/renderer/ShaderLoader.h"

#include "PathTracerShared.h"

#include <filesystem>
#include <vector>

namespace air::pt
{

class VolumeMarcher
{
public:
    explicit VolumeMarcher(rhi::Device& device);
    ~VolumeMarcher();

    VolumeMarcher(const VolumeMarcher&) = delete;
    VolumeMarcher& operator=(const VolumeMarcher&) = delete;

    // Loads VolumeShadows.cs, VolumeProbes.cs and VolumeMarch.cs (their pipelines are created the first
    // time a scene has volumes to march).
    void Initialize(const std::filesystem::path& shaderDir);
    bool IsAvailable() const;

    // Starts a frame: chooses the light groups that light the volumes, renders their moment shadow maps
    // when they are out of date and fills `shadows` (FrameData::volumeShadows).
    void BeginFrame(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const VolumeMarchSettings& settings,
                    gpu::VolumeShadowsData& shadows);
    // (The trace's surfaces trace their own opaque shadows: they read the volumes' slices only.)
    // What a pass that reads the maps (the trace) declares.
    void DeclareShadowReads(rdg::PassBuilder& builder) const;
    // The probes (with the trace's scene bindings), then the march of the trace's samples (`surface`)
    // into `accumulation`.
    void AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const gpu::TraceConstants& trace,
                   rdg::TextureRef surface, rdg::TextureRef accumulation, u32 width, u32 height, u32 samplesPerPixel,
                   u32 debugView, bool reset, bool accumulating, bool furnace);

    u32 GetLightCount() const { return u32(mLights.size()); }
    u32 GetShadowRenders() const { return mShadowRenders; } // times the maps were rendered

private:
    void ReleaseShadows();

    rhi::Device& mDevice;
    renderer::LazyComputePipeline mShadowPipeline;
    renderer::LazyComputePipeline mProbePipeline;
    renderer::LazyComputePipeline mMarchPipeline;
    rhi::BufferHandle mBlueNoise; // StructuredBuffer<uint>, 64 x 64 ranks

    // The moment shadow maps: one slice per map (the sun's first, then the light groups').
    rhi::TextureHandle mB0;
    rhi::TextureHandle mMoments;
    rhi::TextureHandle mOccluders; // the depth of each texel ray's first opaque hit
    rhi::Access mMapsAccess = rhi::Access::None;
    u32 mShadowSize = 0;
    rhi::BufferHandle mMaps;        // StructuredBuffer<MomentShadowMap>
    rhi::BufferHandle mGroupMaps;   // StructuredBuffer<uint>: per light group its map
    rhi::BufferHandle mMarchLights; // StructuredBuffer<uint>: the groups with maps
    std::vector<air::gpu::MomentShadowMap> mMapData;
    std::vector<u32> mLights; // the light groups with maps
    u32 mSunMap = air::gpu::kInvalidIndex;
    u64 mShadowVersion = ~0ull; // the scene version the maps show
    VolumeMarchSettings mShadowSettings;
    u32 mShadowRenders = 0;

    // The probe field (VolumeMarchShared.h): kVolumeProbes per volume.
    rhi::TextureHandle mProbes;
    rhi::Access mProbesAccess = rhi::Access::None;
    u32 mProbeVolumes = 0;
    u64 mProbeVersion = ~0ull;
    u32 mProbeFrames = 0; // blended into them since they were reset
    u32 mFrame = 0;

    // This frame, from BeginFrame.
    VolumeMarchSettings mSettings;
    bool mRenderShadows = false;
    rdg::TextureRef mB0Ref;
    rdg::TextureRef mMomentsRef;
    rdg::TextureRef mOccludersRef;
};

} // namespace air::pt
