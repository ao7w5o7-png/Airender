#include "CloudMarcher.h"

#include "air/core/Assert.h"

#include "CloudMarchShared.h"

namespace air::pt
{

using rhi::Access;

namespace
{

constexpr Access kRest = Access::SRVCompute;

void Destroy(rhi::Device& device, rhi::TextureHandle& texture)
{
    if (texture.IsValid())
    {
        device.DestroyTexture(texture);
    }
    texture = {};
}

rhi::TextureHandle MakeTexture(rhi::Device& device, rhi::Format format, u32 width, u32 height, const char* name)
{
    return device.CreateTexture({.format = format,
                                 .width = width,
                                 .height = height,
                                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                 .debugName = name});
}

// The layer pixel of every 4 x 4 block marched in frame i (x | y << 2): a 4 x 4 Bayer order, so that
// any 4 consecutive frames cover the block evenly.
constexpr u32 kOffsets[16] = {0u, 10u, 2u, 8u, 5u, 15u, 7u, 13u, 1u, 11u, 3u, 9u, 4u, 14u, 6u, 12u};

} // namespace

CloudMarcher::CloudMarcher(rhi::Device& device) : mDevice(device) {}

CloudMarcher::~CloudMarcher()
{
    Destroy(mDevice, mShadowMap);
    Destroy(mDevice, mSkyMap);
    for (u32 i = 0; i < 2; ++i)
    {
        Destroy(mDevice, mLayers[i]);
        Destroy(mDevice, mDepths[i]);
    }
    for (renderer::LazyComputePipeline* pipeline :
         {&mShadowPipeline, &mSkyPipeline, &mMarchPipeline, &mReconstructPipeline})
    {
        pipeline->Destroy(mDevice);
    }
}

void CloudMarcher::Initialize(const std::filesystem::path& shaderDir)
{
    const u32 bytes = sizeof(gpu::CloudMarchConstants);
    mShadowPipeline.Load(mDevice, shaderDir, "CloudShadow.cs", "Clouds.Shadow", bytes);
    mSkyPipeline.Load(mDevice, shaderDir, "CloudSky.cs", "Clouds.Sky", bytes);
    mMarchPipeline.Load(mDevice, shaderDir, "CloudMarch.cs", "Clouds.March", bytes);
    mReconstructPipeline.Load(mDevice, shaderDir, "CloudReconstruct.cs", "Clouds.Reconstruct", bytes);
}

bool CloudMarcher::IsAvailable() const
{
    return mShadowPipeline.IsAvailable() && mSkyPipeline.IsAvailable() && mMarchPipeline.IsAvailable() &&
           mReconstructPipeline.IsAvailable();
}

void CloudMarcher::BeginFrame(const renderer::GpuScene& scene, u32 width, u32 height, bool historyValid,
                              gpu::FrameData& frame)
{
    if (!mShadowMap.IsValid())
    {
        mShadowMap = MakeTexture(mDevice, rhi::Format::R32_FLOAT, gpu::kCloudShadowMapSize, gpu::kCloudShadowMapSize,
                                 "Clouds.ShadowMap");
        mSkyMap = MakeTexture(mDevice, rhi::Format::RGBA32_FLOAT, gpu::kCloudSkyMapSize, gpu::kCloudSkyMapSize,
                              "Clouds.SkyMap");
        mShadowAccess = Access::None;
        mSkyAccess = Access::None;
        mShadowVersion = ~0ull;
    }
    const u32 layerWidth = width;
    const u32 layerHeight = height;
    if (layerWidth != mLayerWidth || layerHeight != mLayerHeight || !mLayers[0].IsValid())
    {
        for (u32 i = 0; i < 2; ++i)
        {
            Destroy(mDevice, mLayers[i]);
            Destroy(mDevice, mDepths[i]);
            mLayers[i] = MakeTexture(mDevice, rhi::Format::RGBA32_FLOAT, layerWidth, layerHeight,
                                     i == 0 ? "Clouds.Layer[0]" : "Clouds.Layer[1]");
            mDepths[i] = MakeTexture(mDevice, rhi::Format::R32_FLOAT, layerWidth, layerHeight,
                                     i == 0 ? "Clouds.LayerDepth[0]" : "Clouds.LayerDepth[1]");
            mLayersAccess[i] = Access::None;
        }
        mLayerWidth = layerWidth;
        mLayerHeight = layerHeight;
        mHistory = false;
    }
    mRenderShadow = scene.version != mShadowVersion;
    mShadowVersion = scene.version;
    mUseHistory = mHistory && historyValid;
    frame.clouds.shadowMap = mDevice.GetBindlessSRV(mShadowMap);
    frame.clouds.skyMap = mDevice.GetBindlessSRV(mSkyMap);
    frame.cloudLayer = mDevice.GetBindlessSRV(mLayers[mCurrent]);
    frame.cloudDepth = mDevice.GetBindlessSRV(mDepths[mCurrent]);
}

void CloudMarcher::AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene,
                             const CloudMarchSettings& settings, u32 frameData, u32 frameSlot, u32 skyTable,
                             u32 frameIndex)
{
    (void)scene;
    gpu::CloudMarchConstants base{};
    base.frameData = frameData;
    base.frameSlot = frameSlot;
    base.skyTable = skyTable;
    base.layerWidth = mLayerWidth;
    base.layerHeight = mLayerHeight;
    base.offset = kOffsets[frameIndex % 16];
    base.history = air::gpu::kInvalidIndex;
    base.historyDepth = air::gpu::kInvalidIndex;

    mShadowRef = graph.ImportTexture(mShadowMap, mShadowAccess, kRest);
    mShadowAccess = kRest;
    mSkyRef = graph.ImportTexture(mSkyMap, mSkyAccess, kRest);
    mSkyAccess = kRest;
    const u32 current = mCurrent;
    mLayerRef = graph.ImportTexture(mLayers[current], mLayersAccess[current], kRest);
    mDepthRef = graph.ImportTexture(mDepths[current], mLayersAccess[current], kRest);
    mLayersAccess[current] = kRest;
    rdg::TextureRef past;
    rdg::TextureRef pastDepth;
    if (mUseHistory)
    {
        past = graph.ImportTexture(mLayers[current ^ 1u], mLayersAccess[current ^ 1u], kRest);
        pastDepth = graph.ImportTexture(mDepths[current ^ 1u], mLayersAccess[current ^ 1u], kRest);
        mLayersAccess[current ^ 1u] = kRest;
    }
    mCurrent ^= 1u;
    mHistory = true;

    // A pass over a target of width x height (threads over dispatchWidth x dispatchHeight) with `bind`
    // filling the rest of its constants.
    auto add = [&](const char* name, renderer::LazyComputePipeline& lazy, u32 width, u32 height, u32 dispatchWidth,
                   u32 dispatchHeight, std::initializer_list<rdg::TextureRef> reads,
                   std::initializer_list<rdg::TextureRef> writes, auto bind)
    {
        const rhi::PipelineHandle pipeline = lazy.Get(mDevice);
        AIR_CHECK(pipeline.IsValid(), "{} did not compile", name);
        gpu::CloudMarchConstants c = base;
        c.width = width;
        c.height = height;
        c.steps = settings.steps;
        c.lightSamples = settings.lightSamples;
        const std::vector<rdg::TextureRef> readList(reads);
        const std::vector<rdg::TextureRef> writeList(writes);
        graph.AddPass(
            name, rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                for (rdg::TextureRef t : readList)
                {
                    b.Read(t, kRest);
                }
                for (rdg::TextureRef t : writeList)
                {
                    b.Write(t, Access::UAVCompute);
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                bind(ctx, c);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&c, sizeof(c));
                ctx.cmd.Dispatch((dispatchWidth + 7) / 8, (dispatchHeight + 7) / 8, 1);
            });
    };

    if (mRenderShadow)
    {
        ++mShadowRenders;
        const rdg::TextureRef shadow = mShadowRef;
        constexpr u32 size = gpu::kCloudShadowMapSize;
        add("Clouds.Shadow", mShadowPipeline, size, size, size, size, {}, {shadow},
            [=](rdg::PassContext& ctx, gpu::CloudMarchConstants& c) { c.target = ctx.UAV(shadow); });
    }
    const rdg::TextureRef sky = mSkyRef;
    constexpr u32 skySize = gpu::kCloudSkyMapSize;
    add("Clouds.Sky", mSkyPipeline, skySize, skySize, skySize, skySize, {}, {sky},
        [=, steps = settings.skySteps, lightSamples = settings.skyLightSamples](rdg::PassContext& ctx,
                                                                                gpu::CloudMarchConstants& c)
        {
            c.target = ctx.UAV(sky);
            c.steps = steps;
            c.lightSamples = lightSamples;
        });

    const u32 marchWidth = (mLayerWidth + 3) / 4;
    const u32 marchHeight = (mLayerHeight + 3) / 4;
    auto make = [&](rhi::Format format, const char* name)
    {
        return graph.CreateTexture({.format = format,
                                    .width = marchWidth,
                                    .height = marchHeight,
                                    .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                    .debugName = name});
    };
    const rdg::TextureRef marched = make(rhi::Format::RGBA32_FLOAT, "Clouds.Marched");
    const rdg::TextureRef marchedDepth = make(rhi::Format::R32_FLOAT, "Clouds.MarchedDepth");
    add("Clouds.March", mMarchPipeline, marchWidth, marchHeight, marchWidth, marchHeight, {}, {marched, marchedDepth},
        [=](rdg::PassContext& ctx, gpu::CloudMarchConstants& c)
        {
            c.target = ctx.UAV(marched);
            c.targetDepth = ctx.UAV(marchedDepth);
        });

    const rdg::TextureRef layer = mLayerRef;
    const rdg::TextureRef layerDepth = mDepthRef;
    if (past.IsValid())
    {
        add("Clouds.Reconstruct", mReconstructPipeline, marchWidth, marchHeight, mLayerWidth, mLayerHeight,
            {marched, marchedDepth, past, pastDepth}, {layer, layerDepth},
            [=](rdg::PassContext& ctx, gpu::CloudMarchConstants& c)
            {
                c.marched = ctx.SRV(marched);
                c.marchedDepth = ctx.SRV(marchedDepth);
                c.history = ctx.SRV(past);
                c.historyDepth = ctx.SRV(pastDepth);
                c.target = ctx.UAV(layer);
                c.targetDepth = ctx.UAV(layerDepth);
            });
    }
    else
    {
        add("Clouds.Reconstruct", mReconstructPipeline, marchWidth, marchHeight, mLayerWidth, mLayerHeight,
            {marched, marchedDepth}, {layer, layerDepth},
            [=](rdg::PassContext& ctx, gpu::CloudMarchConstants& c)
            {
                c.marched = ctx.SRV(marched);
                c.marchedDepth = ctx.SRV(marchedDepth);
                c.target = ctx.UAV(layer);
                c.targetDepth = ctx.UAV(layerDepth);
            });
    }
}

void CloudMarcher::DeclareReads(rdg::PassBuilder& builder) const
{
    for (rdg::TextureRef t : {mShadowRef, mSkyRef, mLayerRef, mDepthRef})
    {
        builder.Read(t, kRest);
    }
}

} // namespace air::pt
