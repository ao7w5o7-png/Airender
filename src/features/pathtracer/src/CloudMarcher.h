#pragma once

// The real-time clouds (docs/architecture.md §8.19): instead of every path walking through the clouds
// (§8.10: 78 ms for a 960 x 540 frame of `cumulus`), they are ray marched in the manner of Nubis
// (Schneider, Horizon Zero Dawn and Forbidden West) into three maps the path tracer reads
// (kFlagCloudMaps):
//   Clouds.Shadow       the optical depth along the sun from their base up, over one tile of them;
//                       rendered when the clouds or the sun change
//   Clouds.Sky          what they add to and leave of the sky towards each upward direction, seen from
//                       the camera: the light of the rays that leave the scene after a bounce
//   Clouds.March        the clouds over one pixel of every 2 x 2 block of the half-size layer (a
//                       sixteenth of the render's pixels), a different one each frame
//   Clouds.Reconstruct  the layer: the marched pixels blended into their history, the others reprojected
//                       from it along the depth of the clouds' light
// all before PathTracer.Trace, which lays the layer over its camera rays (AddCloudLayer in
// PathTracerKernel.hlsli).

#include "air/features/pathtracer/PathTracer.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/Scene.h"
#include "air/renderer/ShaderLoader.h"

#include "PathTracerShared.h"

#include <array>
#include <filesystem>

namespace air::pt
{

class CloudMarcher
{
public:
    explicit CloudMarcher(rhi::Device& device);
    ~CloudMarcher();

    CloudMarcher(const CloudMarcher&) = delete;
    CloudMarcher& operator=(const CloudMarcher&) = delete;

    // Loads CloudShadow.cs, CloudSky.cs, CloudMarch.cs and CloudReconstruct.cs.
    void Initialize(const std::filesystem::path& shaderDir);
    bool IsAvailable() const;

    // Starts a frame of the given render size: makes the maps (when needed) and points `frame` at them
    // (CloudData::shadowMap, skyMap, FrameData::cloudLayer, cloudDepth).
    void BeginFrame(const renderer::GpuScene& scene, u32 width, u32 height, bool historyValid, gpu::FrameData& frame);
    // The passes, which read the frame's FrameData (with the sky's table).
    void AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const CloudMarchSettings& settings,
                   u32 frameData, u32 frameSlot, u32 skyTable, u32 frameIndex);
    // What a pass that reads the maps (the trace) declares.
    void DeclareReads(rdg::PassBuilder& builder) const;

    u32 GetShadowRenders() const { return mShadowRenders; }

private:
    rhi::Device& mDevice;
    renderer::LazyComputePipeline mShadowPipeline;
    renderer::LazyComputePipeline mSkyPipeline;
    renderer::LazyComputePipeline mMarchPipeline;
    renderer::LazyComputePipeline mReconstructPipeline;

    rhi::TextureHandle mShadowMap; // R32F, kCloudShadowMapSize^2
    rhi::Access mShadowAccess = rhi::Access::None;
    u64 mShadowVersion = ~0ull; // the scene version it shows
    bool mRenderShadow = false;
    u32 mShadowRenders = 0;
    rhi::TextureHandle mSkyMap; // RGBA32F, kCloudSkyMapSize^2
    rhi::Access mSkyAccess = rhi::Access::None;

    // The layer and its depth (half the render size): [mCurrent] is written this frame, the other holds
    // last frame's when mHistory.
    std::array<rhi::TextureHandle, 2> mLayers;
    std::array<rhi::TextureHandle, 2> mDepths;
    std::array<rhi::Access, 2> mLayersAccess{rhi::Access::None, rhi::Access::None};
    u32 mLayerWidth = 0;
    u32 mLayerHeight = 0;
    u32 mCurrent = 0;
    bool mHistory = false;
    bool mUseHistory = false;

    // This frame's, for DeclareReads.
    rdg::TextureRef mShadowRef;
    rdg::TextureRef mSkyRef;
    rdg::TextureRef mLayerRef;
    rdg::TextureRef mDepthRef;
};

} // namespace air::pt
