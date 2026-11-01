// Probe GI (docs/architecture.md §8.21): real-time global illumination without ray tracing hardware. The
// scene is voxelized into cascades of a distance field with surface voxels that keep their light
// (VoxelScene); probes on the visible surfaces, one per tile of pixels, trace their rays over the depth
// buffer, then through the voxels, then end in a world-space radiance cache of probes over the cascades;
// the probes' light as spherical harmonics, filtered among neighbours, gives every pixel its irradiance and
// rough specular light, accumulated over frames, added to the frame (and, under nrd, to its signals).
//
// It consumes slots::kExternalGi: the path tracer then renders the emission and the direct light of the
// emitters and the sun only, and this feature adds the light of the sky and of other surfaces. Run it after
// the path tracer: --features pathtracer,probe_gi,taa,tonemap (pathtracer.debugView=15 accumulates the
// reference of what it adds; probe_gi.view=1 shows that alone).

#include "air/core/Log.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/ShaderLoader.h"

#include "ProbeGiShared.h"
#include "VoxelScene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <optional>

namespace air::pgi
{
namespace
{

using rhi::Access;
namespace slots = renderer::slots;

gpu::GiCamera ToGpu(const renderer::Camera& camera, float aspect)
{
    gpu::GiCamera out{};
    out.position = {camera.position, std::tan(camera.verticalFov * 0.5f)};
    out.forward = {camera.Forward(), aspect};
    out.right = {camera.Right(), 0.0f};
    out.up = {camera.Up(), 0.0f};
    return out;
}

// After a change of the lights every pixel keeps a short history this long, at least kRelightBlend a frame: of
// their light before, 0.92^48 = 2% is left when the slower blends take over.
constexpr u32 kFastHistoryFrames = 48;
constexpr float kRelightBlend = 0.08f;

bool IsVoxelView(u32 view)
{
    return view == gpu::kGiViewVoxels || view == gpu::kGiViewVoxelLight || view == gpu::kGiViewSteps ||
           view == gpu::kGiViewCascades || view == gpu::kGiViewCache;
}

class ProbeGiFeature final : public renderer::RenderFeature
{
public:
    ProbeGiFeature()
    {
        mParameters.Add("enabled", &mEnabled,
                        "Probe GI: the light of the sky and of other surfaces without ray tracing");
        mParameters.Add("view", &mView,
                        "0 lit, 1 the GI alone (pathtracer.debugView=15 is its reference), 2 diffuse irradiance / pi, "
                        "3 specular light, 4 voxels' albedo, 5 voxels' light, 6 voxel trace steps (debugScale), 7 the "
                        "cascade rays hit in, 8 screen probes, 9 the radiance cache's irradiance where camera rays hit",
                        0, gpu::kGiViewCount - 1);
        mParameters.Group("Screen probes");
        mParameters.Add("probeTile", &mProbeTile, "Pixels per side of a screen probe's tile", 4, 64);
        mParameters.Add(
            "adaptiveProbes", &mAdaptive,
            "Probes of their own where the tiles' probes around a pixel fit nothing (up to half as many again)");
        mParameters.Add("reflections", &mReflections,
                        "Smooth surfaces trace their own reflections (a ray per 2 x 2 pixels below roughness 0.4)");
        mParameters.Add("rayFilterRadius", &mRayFilterRadius,
                        "Screen probes' rays averaged per direction with neighbours this many probes away", 0, 3);
        mParameters.Add("filterRadius", &mFilterRadius, "Screen probes averaged with neighbours this many probes away",
                        0, 3);
        mParameters.Add("probeBlend", &mProbeBlend,
                        "Least weight of a frame in a screen probe's history (its harmonics over frames; 1: none)",
                        0.01f, 1.0f);
        mParameters.Add("temporalBlend", &mTemporalBlend,
                        "Least weight of a frame in the history of a pixel that moves (a pixel or more)", 0.01f, 1.0f);
        mParameters.Add("staticBlend", &mStaticBlend,
                        "Least weight of a frame in the history of a pixel that does not move (8% or temporalBlend "
                        "for a while after the lights change)",
                        0.01f, 1.0f);
        mParameters.Add("screenTraces", &mScreenTraces, "Probe rays march the depth buffer before the voxels");
        mParameters.Add("screenSteps", &mScreenSteps, "Samples of a screen trace", 1, 64);
        mParameters.Add("screenThickness", &mScreenThickness,
                        "Relative depth behind a pixel's surface a screen trace takes as a hit", 0.001f, 1.0f);
        mParameters.Group("Voxels");
        mParameters.Add("cascades", &mVoxel.cascades, "Voxel cascades around the camera (128^3 voxels each)", 1,
                        gpu::kMaxCascades);
        mParameters.Add("voxelSize", &mVoxel.voxelSize, "Metres a voxel of the finest cascade spans", 0.005f, 10.0f);
        mParameters.Add("voxelCapacity", &mVoxel.voxelCapacity,
                        "Surface voxels a cascade holds at first (doubled when they do not fit)", 1024, 1u << 21);
        mParameters.Add("workCapacity", &mVoxel.workCapacity,
                        "(triangle, brick) work items of a voxelization at first (doubled when they do not fit)", 1024,
                        1u << 25);
        mParameters.Add("rebuildsPerFrame", &mVoxel.rebuildsPerFrame,
                        "Cascades voxelized in one frame when several moved", 1, gpu::kMaxCascades);
        mParameters.Add("scroll", &mVoxel.scroll,
                        "A cascade that moves voxelizes only what it did not cover (off: all of it again)");
        mParameters.Group("Voxel lighting");
        mParameters.Add("lightingBudget", &mVoxel.lightingBudget,
                        "Surface voxels of a cascade lit per frame (a budget that cycles through them all)", 64,
                        1u << 22);
        mParameters.Add("lightingBlend", &mVoxel.lightingBlend, "Weight of a new estimate in a surface voxel's light",
                        0.01f, 1.0f);
        mParameters.Add("radiosityRays", &mVoxel.radiosityRays,
                        "Rays per surface voxel face and lighting that gather its indirect light (0: the radiance "
                        "cache's irradiance)",
                        0, 16);
        mParameters.Add("radiosityDistance", &mVoxel.radiosityDistance,
                        "Voxels a radiosity ray is traced before the radiance cache gives the rest (0: to the end)", 0,
                        1024);
        mParameters.Add("multiBounce", &mMultiBounce,
                        "The voxels reflect the light of other voxels and the sky too (off: their direct light only, "
                        "one bounce of GI where screen traces bring no more; to check it)");
        mParameters.Group("Radiance cache");
        mParameters.Add("cacheProbes", &mVoxel.cacheProbesPerFrame,
                        "Radiance cache probes traced per level and frame (of 4096)", 1, gpu::kCacheProbes);
        mParameters.Add("cacheBlend", &mVoxel.cacheBlend, "Weight of a new trace in a radiance cache probe", 0.01f,
                        1.0f);
        mParameters.Add("cacheDistance", &mCacheDistance,
                        "Probe rays end in the radiance cache this many cache spacings out (0: they trace the voxels "
                        "to the end, and the cache lights only pixels without a screen probe)",
                        0.0f, 64.0f);
        mParameters.Group("Debug");
        mParameters.Add("debugScale", &mDebugScale, "Full scale of the steps heatmap", 1.0f, 1e4f);
    }

    ~ProbeGiFeature() override
    {
        if (!mDevice)
        {
            return;
        }
        mDevice->WaitIdle();
        mVoxels.reset();
        for (rhi::TextureHandle t :
             {mReflected, mIrradiance[0], mIrradiance[1], mSpecular[0], mSpecular[1], mSurface[0], mSurface[1]})
        {
            if (t.IsValid())
            {
                mDevice->DestroyTexture(t);
            }
        }
        DestroyProbeHistory();
        for (renderer::LazyComputePipeline* p :
             {&mView3d, &mPlace, &mAdapt, &mTrace, &mRayFilter, &mSh, &mProbeTemporal, &mFilter, &mReflectionTrace,
              &mReflectionResolve, &mGather, &mTemporal, &mComposite})
        {
            p->Destroy(*mDevice);
        }
    }

    bool Initialize(const renderer::FeatureInitContext& context) override
    {
        mDevice = &context.device;
        mWidth = context.renderWidth;
        mHeight = context.renderHeight;
        mVoxels = std::make_unique<VoxelScene>(context.device);
        if (!mVoxels->Initialize(context.shaderDir))
        {
            return false;
        }
        const u32 gi = sizeof(gpu::GiConstants);
        mView3d.Load(context.device, context.shaderDir, "VoxelView.cs", "ProbeGi.VoxelView", gi);
        mPlace.Load(context.device, context.shaderDir, "ProbePlace.cs", "ProbeGi.Probes.Place", gi);
        mAdapt.Load(context.device, context.shaderDir, "ProbeAdapt.cs", "ProbeGi.Probes.Adapt", gi);
        mTrace.Load(context.device, context.shaderDir, "ProbeTrace.cs", "ProbeGi.Probes.Trace", gi);
        mRayFilter.Load(context.device, context.shaderDir, "ProbeRayFilter.cs", "ProbeGi.Probes.RayFilter", gi);
        mSh.Load(context.device, context.shaderDir, "ProbeSh.cs", "ProbeGi.Probes.Sh", gi);
        mProbeTemporal.Load(context.device, context.shaderDir, "ProbeTemporal.cs", "ProbeGi.Probes.Temporal", gi);
        mFilter.Load(context.device, context.shaderDir, "ProbeFilter.cs", "ProbeGi.Probes.Filter", gi);
        mReflectionTrace.Load(context.device, context.shaderDir, "ReflectionTrace.cs", "ProbeGi.Reflections.Trace", gi);
        mReflectionResolve.Load(context.device, context.shaderDir, "ReflectionResolve.cs",
                                "ProbeGi.Reflections.Resolve", gi);
        mGather.Load(context.device, context.shaderDir, "ProbeGather.cs", "ProbeGi.Gather", gi);
        mTemporal.Load(context.device, context.shaderDir, "GiTemporal.cs", "ProbeGi.Temporal", gi);
        mComposite.Load(context.device, context.shaderDir, "GiComposite.cs", "ProbeGi.Composite", gi);
        for (const renderer::LazyComputePipeline* p :
             {&mView3d, &mPlace, &mAdapt, &mTrace, &mRayFilter, &mSh, &mProbeTemporal, &mFilter, &mReflectionTrace,
              &mReflectionResolve, &mGather, &mTemporal, &mComposite})
        {
            if (!p->IsAvailable())
            {
                AIR_LOG(Error, "ProbeGi", "shaders are missing from {}", context.shaderDir.string());
                return false;
            }
        }
        auto make = [&](const char* name)
        {
            return context.device.CreateTexture(
                {.format = rhi::Format::RGBA16_FLOAT,
                 .width = mWidth,
                 .height = mHeight,
                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                 .debugName = name});
        };
        mReflected = make("ProbeGi.Reflected");
        mIrradiance[0] = make("ProbeGi.Irradiance[0]");
        mIrradiance[1] = make("ProbeGi.Irradiance[1]");
        mSpecular[0] = make("ProbeGi.Specular[0]");
        mSpecular[1] = make("ProbeGi.Specular[1]");
        mSurface[0] = make("ProbeGi.Surface[0]");
        mSurface[1] = make("ProbeGi.Surface[1]");
        return true;
    }

    renderer::FeatureStage GetStage() const override { return renderer::FeatureStage::Scene; }

    std::vector<std::string_view> ConsumedSlots() const override
    {
        if (!mEnabled)
        {
            return {};
        }
        return {slots::kExternalGi,    slots::kNormalRoughness, slots::kViewDepth, slots::kMotionVectors,
                slots::kDiffuseAlbedo, slots::kSpecularAlbedo,  slots::kEmission};
    }

    json::Value GetStatus() const override
    {
        json::Value status = json::Value::MakeObject();
        status["active"] = mActive;
        status["cascades"] = mVoxels ? mVoxels->GetCascadeCount() : 0u;
        status["builtCascades"] = mVoxels ? mVoxels->GetBuiltCascades() : 0u;
        status["rebuilds"] = mVoxels ? mVoxels->GetRebuilds() : 0u;
        status["fullRebuilds"] = mVoxels ? mVoxels->GetFullRebuilds() : 0u;
        status["relights"] = mVoxels ? mVoxels->GetRelights() : 0u;
        status["probes"] = mProbesX * mProbesY;
        if (mVoxels)
        {
            json::Value cascades = json::Value::MakeArray();
            for (const auto& counters : mVoxels->ReadCounters())
            {
                json::Value c = json::Value::MakeObject();
                const u32 freeSlots = counters[gpu::kCounterFreeTail / 4] - counters[gpu::kCounterFreeHead / 4];
                c["workItems"] = counters[gpu::kCounterWorkItems / 4];
                c["surfaceVoxels"] = counters[gpu::kCounterVoxels / 4] - freeSlots;
                c["freeSlots"] = freeSlots;
                c["workOverflow"] = counters[gpu::kCounterWorkOverflow / 4];
                c["voxelOverflow"] = counters[gpu::kCounterVoxelOverflow / 4];
                cascades.Push(std::move(c));
            }
            status["voxelization"] = std::move(cascades);
        }
        return status;
    }

    void AddPasses(renderer::FrameContext& context) override
    {
        mActive = false;
        if (!mEnabled)
        {
            return;
        }
        const renderer::Blackboard& bb = context.blackboard;
        const auto depth = bb.FindTexture(slots::kViewDepth);
        const auto normal = bb.FindTexture(slots::kNormalRoughness);
        const auto motion = bb.FindTexture(slots::kMotionVectors);
        const auto diffuse = bb.FindTexture(slots::kDiffuseAlbedo);
        const auto specular = bb.FindTexture(slots::kSpecularAlbedo);
        const auto emission = bb.FindTexture(slots::kEmission);
        if (!depth || !normal || !motion || !diffuse || !specular || bb.GetValue(slots::kOutputIsDisplay) != 0.0)
        {
            mHistoryValid = false;
            return; // no surfaces to light (a debug view of the path tracer, or it did not run)
        }
        const bool external = bb.GetValue(slots::kExternalGi) != 0.0;
        if (!external && !IsVoxelView(mView))
        {
            mHistoryValid = false;
            return; // the frame has its indirect light already
        }
        mActive = true;
        rdg::RenderGraph& graph = context.graph;
        const renderer::GpuScene& scene = context.scene;
        const bool furnace = bb.GetValue(slots::kFurnace) != 0.0;
        const bool reset = bb.GetValue(slots::kHistoryReset) != 0.0 || !mHistoryValid;

        // The frame's data.
        const float aspect = float(mWidth) / float(mHeight);
        gpu::GiFrame frame{};
        frame.camera = ToGpu(context.camera, aspect);
        frame.previousCamera = ToGpu(mPreviousCamera.value_or(context.camera), aspect);
        mPreviousCamera = context.camera;
        frame.sky = scene.sky;
        if (scene.aerialInscatter.IsValid())
        {
            frame.sky.aerialInscatter = mDevice->GetBindlessSRV(scene.aerialInscatter);
            frame.sky.aerialTransmittance = mDevice->GetBindlessSRV(scene.aerialTransmittance);
        }
        else
        {
            frame.sky.aerialDistance = 0.0f;
        }
        if (furnace)
        {
            // The path tracer's furnace: a uniform sky (white for a physical one), no sun, no air.
            const hlsl::float3 uniform = scene.sky.width == 0 ? scene.sky.uniformRadiance : hlsl::float3(1.0f);
            const float luminance = hlsl::dot(uniform, hlsl::float3(0.2126f, 0.7152f, 0.0722f));
            frame.sky.width = 0;
            frame.sky.height = 0;
            frame.sky.uniformRadiance = luminance > 0.0f ? uniform : hlsl::float3(1.0f);
            frame.sky.sunRadiance = hlsl::float3(0.0f);
            frame.sky.aerialDistance = 0.0f;
        }
        frame.jitter = {float(bb.GetValue(slots::kJitterX)), float(bb.GetValue(slots::kJitterY))};
        frame.previousJitter = mPreviousJitter;
        mPreviousJitter = frame.jitter;
        frame.width = mWidth;
        frame.height = mHeight;
        frame.frameIndex = mFrameIndex++;
        frame.exposure = float(bb.GetValue(slots::kExposure, renderer::ExposureScale(context.camera.ev100)));
        frame.view = mView;
        frame.flags = (furnace ? gpu::kGiFurnace : 0u) | (reset ? 0u : gpu::kGiHistory) |
                      (mScreenTraces ? gpu::kGiScreenTraces : 0u) | (mCacheDistance > 0.0f ? gpu::kGiCache : 0u) |
                      (mMultiBounce ? 0u : gpu::kGiDirectVoxels);
        frame.lightCount = furnace ? 0u : scene.lightCount;
        frame.lights = mDevice->GetBindlessSRV(scene.lights);
        frame.skyTable = mDevice->GetBindlessSRV(scene.skyTable);
        mProbesX = (mWidth + mProbeTile - 1) / mProbeTile;
        mProbesY = (mHeight + mProbeTile - 1) / mProbeTile;
        frame.probeTile = mProbeTile;
        frame.probesX = mProbesX;
        frame.probesY = mProbesY;
        mAdaptiveRows = mAdaptive ? (mProbesY + 1) / 2 : 0; // up to half as many probes again
        frame.adaptiveRows = mAdaptiveRows;
        frame.cacheDistance = mCacheDistance;
        frame.screenThickness = mScreenThickness;
        frame.screenSteps = mScreenSteps;
        frame.voxelBlend = mVoxel.lightingBlend;
        frame.cacheBlend = mVoxel.cacheBlend;
        frame.temporalBlend = mTemporalBlend;
        frame.debugScale = mDebugScale;
        frame.probeBlend = mProbeBlend;
        mVoxels->BeginFrame(scene, context.camera, mVoxel, furnace, frame);
        // The still pixels' longer history would follow a change of the lights over a hundred frames: for a
        // while after one they keep the moving pixels' (the voxels and cache are lit again at once).
        if (mVoxels->GetRelights() != mRelightsSeen)
        {
            mRelightsSeen = mVoxels->GetRelights();
            mFastHistoryFrames = kFastHistoryFrames;
        }
        const float relightBlend = std::max(mTemporalBlend, kRelightBlend);
        frame.temporalBlend = mFastHistoryFrames > 0 ? relightBlend : mTemporalBlend;
        frame.staticBlend = mFastHistoryFrames > 0 ? relightBlend : std::min(mStaticBlend, mTemporalBlend);
        mFastHistoryFrames -= mFastHistoryFrames > 0 ? 1u : 0u;
        // Every pass reads the frame's constants in place, from the frame's upload memory.
        const rdg::UploadAllocator::Allocation frameData = graph.AllocateUpload(sizeof(frame), 16);
        std::memcpy(frameData.cpu, &frame, sizeof(frame));

        mVoxels->AddPasses(graph, scene, frameData.srv, u32(frameData.offset), frame.frameIndex);

        gpu::GiConstants base{};
        base.frameData = frameData.srv;
        base.frameOffset = u32(frameData.offset);
        base.width = mWidth;
        base.height = mHeight;
        const rdg::TextureRef sceneColor = context.sceneColor;
        const u32 width = mWidth;
        const u32 height = mHeight;
        if (IsVoxelView(mView))
        {
            const rhi::PipelineHandle pipeline = mView3d.Get(*mDevice);
            graph.AddPass(
                "ProbeGi.VoxelView", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    mVoxels->DeclareTraceReads(b);
                    b.Write(sceneColor, Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    base.output = ctx.UAV(sceneColor);
                    ctx.cmd.SetPipeline(pipeline);
                    ctx.cmd.PushConstants(&base, sizeof(base));
                    ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
                });
            const bool display =
                mView == gpu::kGiViewVoxels || mView == gpu::kGiViewSteps || mView == gpu::kGiViewCascades;
            context.blackboard.SetValue(slots::kOutputIsDisplay, display ? 1.0 : 0.0);
            context.blackboard.SetValue(slots::kHistoryReset, 1.0);
            mHistoryValid = false;
            mLastView = mView;
            return;
        }

        // Screen probes, and their harmonics over frames: ping-pong buffers, last frame's the history.
        const u32 probes = mProbesX * (mProbesY + mAdaptiveRows); // the tiles' and the adaptive ones
        if (probes != mProbeCount || mProbeTile != mProbeCountTile)
        {
            DestroyProbeHistory();
            mProbeCount = probes;
            mProbeCountTile = mProbeTile;
            for (u32 i = 0; i < 2; ++i)
            {
                mProbes[i] = mDevice->CreateBuffer(
                    {.size = u64(probes) * sizeof(gpu::ScreenProbe),
                     .stride = sizeof(gpu::ScreenProbe),
                     .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                     .debugName = std::format("ProbeGi.Probes[{}]", i)});
                mProbeSh[i] = mDevice->CreateBuffer(
                    {.size = u64(probes) * sizeof(gpu::ProbeSh),
                     .stride = sizeof(gpu::ProbeSh),
                     .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                     .debugName = std::format("ProbeGi.ProbeShHistory[{}]", i)});
            }
            mProbeHistoryValid = false;
        }
        const u32 probeCurrent = mProbeCurrent;
        mProbeCurrent ^= 1u;
        auto importBuffer = [&](rhi::BufferHandle b, Access& access)
        {
            const rdg::BufferRef ref = graph.ImportBuffer(b, access, Access::SRVCompute);
            access = Access::SRVCompute;
            return ref;
        };
        const rdg::BufferRef probeData = importBuffer(mProbes[probeCurrent], mProbesAccess[probeCurrent]);

        const rdg::BufferRef accumulated = importBuffer(mProbeSh[probeCurrent], mProbeShAccess[probeCurrent]);
        const bool probeHistory = mProbeHistoryValid && !reset;
        rdg::BufferRef probesPast;
        rdg::BufferRef accumulatedPast;

        if (probeHistory)
        {
            probesPast = importBuffer(mProbes[probeCurrent ^ 1u], mProbesAccess[probeCurrent ^ 1u]);
            accumulatedPast = importBuffer(mProbeSh[probeCurrent ^ 1u], mProbeShAccess[probeCurrent ^ 1u]);
        }
        mProbeHistoryValid = true;
        const rdg::TextureRef rays =
            graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                 .width = mProbesX * gpu::kScreenProbeTexels,
                                 .height = (mProbesY + mAdaptiveRows) * gpu::kScreenProbeTexels,
                                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                 .debugName = "ProbeGi.ProbeRays"});
        auto shBuffer = [&](const char* name)
        {
            return graph.CreateBuffer({.size = u64(probes) * sizeof(gpu::ProbeSh),
                                       .stride = sizeof(gpu::ProbeSh),
                                       .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                       .debugName = name});
        };
        const rdg::BufferRef harmonics = shBuffer("ProbeGi.ProbeSh");
        const rdg::BufferRef filtered = mFilterRadius > 0 ? shBuffer("ProbeGi.ProbeShFiltered") : rdg::BufferRef{};
        auto pixelTexture = [&](const char* name)
        {
            return graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                        .width = width,
                                        .height = height,
                                        .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                        .debugName = name});
        };
        const rdg::TextureRef irradianceNow = pixelTexture("ProbeGi.IrradianceNow");
        const rdg::TextureRef specularNow = pixelTexture("ProbeGi.SpecularNow");
        const rdg::TextureRef depthRef = *depth;
        const rdg::TextureRef normalRef = *normal;
        const rdg::TextureRef motionRef = *motion;
        const rdg::TextureRef diffuseRef = *diffuse;
        const rdg::TextureRef specularRef = *specular;
        const rdg::TextureRef emissionRef = emission.value_or(rdg::TextureRef{});
        const auto transmittance = bb.FindTexture(slots::kSurfaceTransmittance);
        const rdg::TextureRef transmittanceRef = transmittance.value_or(rdg::TextureRef{});
        // Last frame's reflected light, for the screen traces.
        const rdg::TextureRef reflected = graph.ImportTexture(mReflected, mReflectedAccess, Access::SRVCompute);
        mReflectedAccess = Access::SRVCompute;
        const bool reflectedValid = mHistoryValid && !reset;
        // The pixels' light over frames: ping-pong histories. Last frame's surfaces (normal, view depth) also
        // tell the screen traces whether the image's light at a hit is that surface's.
        const u32 current = mCurrent;
        mCurrent ^= 1u;
        auto import = [&](rhi::TextureHandle t, Access& access)
        {
            const rdg::TextureRef ref = graph.ImportTexture(t, access, Access::SRVCompute);
            access = Access::SRVCompute;
            return ref;
        };
        const rdg::TextureRef irradianceOut = import(mIrradiance[current], mIrradianceAccess[current]);
        const rdg::TextureRef specularOut = import(mSpecular[current], mSpecularAccess[current]);
        const rdg::TextureRef surfaceOut = import(mSurface[current], mSurfaceAccess[current]);
        rdg::TextureRef irradiancePast;
        rdg::TextureRef specularPast;
        rdg::TextureRef surfacePast;
        const bool history = !reset;
        if (history)
        {
            irradiancePast = import(mIrradiance[current ^ 1u], mIrradianceAccess[current ^ 1u]);
            specularPast = import(mSpecular[current ^ 1u], mSpecularAccess[current ^ 1u]);
            surfacePast = import(mSurface[current ^ 1u], mSurfaceAccess[current ^ 1u]);
        }
        const bool surfaceHistory = history && reflectedValid;

        // Adaptive probes (ProbeAdapt.hlsl): their count, and per tile the ones placed in it.
        rdg::BufferRef adaptiveCount;
        rdg::BufferRef adaptiveHeaders;
        if (mAdaptiveRows > 0)
        {
            adaptiveCount =
                graph.CreateBuffer({.size = 16,
                                    .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                    .debugName = "ProbeGi.AdaptiveCount"});
            adaptiveHeaders =
                graph.CreateBuffer({.size = u64(mProbesX) * mProbesY * 16,
                                    .stride = 16,
                                    .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                    .debugName = "ProbeGi.AdaptiveProbes"});
        }
        const rhi::PipelineHandle place = mPlace.Get(*mDevice);
        const u32 probesX = mProbesX;
        const u32 probesY = mProbesY;
        const u32 probeRows = mProbesY + mAdaptiveRows;
        graph.AddPass(
            "ProbeGi.Probes.Place", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(depthRef, Access::SRVCompute);
                b.Read(normalRef, Access::SRVCompute);
                b.Write(probeData, Access::UAVCompute);
                if (adaptiveCount.IsValid())
                {
                    b.Write(adaptiveCount, Access::UAVCompute);
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input0 = ctx.SRV(depthRef);
                base.input1 = ctx.SRV(normalRef);
                base.output = ctx.UAV(probeData);
                base.counters = adaptiveCount.IsValid() ? ctx.UAV(adaptiveCount) : air::gpu::kInvalidIndex;
                ctx.cmd.SetPipeline(place);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch((probesX + 7) / 8, (probeRows + 7) / 8, 1);
            });
        if (adaptiveCount.IsValid())
        {
            const rhi::PipelineHandle adapt = mAdapt.Get(*mDevice);
            graph.AddPass(
                "ProbeGi.Probes.Adapt", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    b.Read(depthRef, Access::SRVCompute);
                    b.Read(normalRef, Access::SRVCompute);
                    b.Write(probeData, Access::UAVCompute);
                    b.Write(adaptiveHeaders, Access::UAVCompute);
                    b.Write(adaptiveCount, Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    base.input0 = ctx.SRV(depthRef);
                    base.input1 = ctx.SRV(normalRef);
                    base.output = ctx.UAV(probeData);
                    base.input5 = ctx.UAV(adaptiveHeaders);
                    base.counters = ctx.UAV(adaptiveCount);
                    ctx.cmd.SetPipeline(adapt);
                    ctx.cmd.PushConstants(&base, sizeof(base));
                    ctx.cmd.Dispatch((probesX + 7) / 8, (probesY + 7) / 8, 1);
                });
        }
        const rhi::PipelineHandle trace = mTrace.Get(*mDevice);
        graph.AddPass(
            "ProbeGi.Probes.Trace", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                mVoxels->DeclareTraceReads(b);
                b.Read(depthRef, Access::SRVCompute);
                b.Read(motionRef, Access::SRVCompute);
                b.Read(reflected, Access::SRVCompute);
                b.Read(probeData, Access::SRVCompute);
                if (surfaceHistory)
                {
                    b.Read(surfacePast, Access::SRVCompute);
                }
                b.Write(rays, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input0 = ctx.SRV(depthRef);
                base.input1 = ctx.SRV(motionRef);
                base.input2 = surfaceHistory ? ctx.SRV(reflected) : air::gpu::kInvalidIndex;
                base.input3 = ctx.SRV(probeData);
                base.input4 = surfaceHistory ? ctx.SRV(surfacePast) : air::gpu::kInvalidIndex;
                base.rays = ctx.UAV(rays);
                ctx.cmd.SetPipeline(trace);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(probesX, probeRows, 1);
            });
        // The rays averaged with the neighbouring probes', per direction.
        const rdg::TextureRef raysFiltered =
            graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                 .width = mProbesX * gpu::kScreenProbeTexels,
                                 .height = (mProbesY + mAdaptiveRows) * gpu::kScreenProbeTexels,
                                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                 .debugName = "ProbeGi.ProbeRaysFiltered"});
        const rhi::PipelineHandle rayFilter = mRayFilter.Get(*mDevice);
        const u32 rayRadius = mRayFilterRadius;
        graph.AddPass(
            "ProbeGi.Probes.RayFilter", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(probeData, Access::SRVCompute);
                b.Read(rays, Access::SRVCompute);
                b.Write(raysFiltered, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input3 = ctx.SRV(probeData);
                base.input4 = ctx.SRV(rays);
                base.input5 = rayRadius;
                base.rays = ctx.UAV(raysFiltered);
                ctx.cmd.SetPipeline(rayFilter);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(probesX, probeRows, 1);
            });
        const rhi::PipelineHandle sh = mSh.Get(*mDevice);
        graph.AddPass(
            "ProbeGi.Probes.Sh", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(probeData, Access::SRVCompute);
                b.Read(raysFiltered, Access::SRVCompute);
                b.Write(harmonics, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input3 = ctx.SRV(probeData);
                base.rays = ctx.SRV(raysFiltered);
                base.output = ctx.UAV(harmonics);
                ctx.cmd.SetPipeline(sh);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(probesX, probeRows, 1);
            });
        const rhi::PipelineHandle probeTemporal = mProbeTemporal.Get(*mDevice);
        graph.AddPass(
            "ProbeGi.Probes.Temporal", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(probeData, Access::SRVCompute);
                b.Read(harmonics, Access::SRVCompute);
                if (probeHistory)
                {
                    b.Read(probesPast, Access::SRVCompute);
                    b.Read(accumulatedPast, Access::SRVCompute);
                }
                b.Write(accumulated, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input3 = ctx.SRV(probeData);
                base.input4 = ctx.SRV(harmonics);
                base.input5 = probeHistory ? ctx.SRV(probesPast) : air::gpu::kInvalidIndex;
                base.input6 = probeHistory ? ctx.SRV(accumulatedPast) : air::gpu::kInvalidIndex;
                base.output = ctx.UAV(accumulated);
                ctx.cmd.SetPipeline(probeTemporal);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch((probesX + 7) / 8, (probeRows + 7) / 8, 1);
            });
        // The harmonics averaged with the neighbouring probes' (off by default: the rays' per-direction filter keeps
        // the occlusion that averaging harmonics blurs; cornell_box's 16 x 16 blocks within 10% of the reference 58%
        // -> 67%).
        const rhi::PipelineHandle filter = mFilter.Get(*mDevice);
        const u32 radius = mFilterRadius;
        const rdg::BufferRef gathered = radius > 0 ? filtered : accumulated;
        if (radius > 0)
        {
            graph.AddPass(
                "ProbeGi.Probes.Filter", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    b.Read(probeData, Access::SRVCompute);
                    b.Read(accumulated, Access::SRVCompute);
                    b.Write(filtered, Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    base.input3 = ctx.SRV(probeData);
                    base.input4 = ctx.SRV(accumulated);
                    base.input5 = radius;
                    base.output = ctx.UAV(filtered);
                    ctx.cmd.SetPipeline(filter);
                    ctx.cmd.PushConstants(&base, sizeof(base));
                    ctx.cmd.Dispatch((probesX + 7) / 8, (probeRows + 7) / 8, 1);
                });
        }
        // Smooth surfaces' own reflection rays: a ray per 2 x 2 pixels, resolved at full resolution
        // (ReflectionCommon.hlsli).
        rdg::TextureRef reflections;
        if (mReflections)
        {
            const u32 halfWidth = (width + 1) / 2;
            const u32 halfHeight = (height + 1) / 2;
            const rdg::TextureRef reflectionLight =
                graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                     .width = halfWidth,
                                     .height = halfHeight,
                                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                     .debugName = "ProbeGi.ReflectionRays"});
            const rdg::TextureRef reflectionDirections =
                graph.CreateTexture({.format = rhi::Format::R32_UINT,
                                     .width = halfWidth,
                                     .height = halfHeight,
                                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                     .debugName = "ProbeGi.ReflectionDirections"});
            reflections = pixelTexture("ProbeGi.Reflections");
            const rhi::PipelineHandle reflectionTrace = mReflectionTrace.Get(*mDevice);
            graph.AddPass(
                "ProbeGi.Reflections.Trace", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    mVoxels->DeclareTraceReads(b);
                    b.Read(depthRef, Access::SRVCompute);
                    b.Read(motionRef, Access::SRVCompute);
                    b.Read(normalRef, Access::SRVCompute);
                    b.Read(reflected, Access::SRVCompute);
                    if (surfaceHistory)
                    {
                        b.Read(surfacePast, Access::SRVCompute);
                    }
                    b.Write(reflectionLight, Access::UAVCompute);
                    b.Write(reflectionDirections, Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    base.input0 = ctx.SRV(depthRef);
                    base.input1 = ctx.SRV(motionRef);
                    base.input2 = surfaceHistory ? ctx.SRV(reflected) : air::gpu::kInvalidIndex;
                    base.input3 = ctx.SRV(normalRef);
                    base.input4 = surfaceHistory ? ctx.SRV(surfacePast) : air::gpu::kInvalidIndex;
                    base.rays = ctx.UAV(reflectionLight);
                    base.output = ctx.UAV(reflectionDirections);
                    ctx.cmd.SetPipeline(reflectionTrace);
                    ctx.cmd.PushConstants(&base, sizeof(base));
                    ctx.cmd.Dispatch((halfWidth + 7) / 8, (halfHeight + 7) / 8, 1);
                });
            const rhi::PipelineHandle reflectionResolve = mReflectionResolve.Get(*mDevice);
            graph.AddPass(
                "ProbeGi.Reflections.Resolve", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    b.Read(depthRef, Access::SRVCompute);
                    b.Read(normalRef, Access::SRVCompute);
                    b.Read(reflectionLight, Access::SRVCompute);
                    b.Read(reflectionDirections, Access::SRVCompute);
                    b.Write(reflections, Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    base.input0 = ctx.SRV(depthRef);
                    base.input1 = ctx.SRV(normalRef);
                    base.input2 = ctx.SRV(reflectionLight);
                    base.input3 = ctx.SRV(reflectionDirections);
                    base.output = ctx.UAV(reflections);
                    ctx.cmd.SetPipeline(reflectionResolve);
                    ctx.cmd.PushConstants(&base, sizeof(base));
                    ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
                });
        }
        const rhi::PipelineHandle gather = mGather.Get(*mDevice);
        graph.AddPass(
            "ProbeGi.Gather", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                mVoxels->DeclareTraceReads(b);
                b.Read(depthRef, Access::SRVCompute);
                b.Read(normalRef, Access::SRVCompute);
                b.Read(probeData, Access::SRVCompute);
                b.Read(gathered, Access::SRVCompute);
                b.Read(raysFiltered, Access::SRVCompute);
                if (reflections.IsValid())
                {
                    b.Read(reflections, Access::SRVCompute);
                }
                if (adaptiveHeaders.IsValid())
                {
                    b.Read(adaptiveHeaders, Access::SRVCompute);
                }
                b.Write(irradianceNow, Access::UAVCompute);
                b.Write(specularNow, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input0 = ctx.SRV(depthRef);
                base.input1 = ctx.SRV(normalRef);
                base.input3 = ctx.SRV(probeData);
                base.input4 = ctx.SRV(gathered);
                base.input2 = ctx.SRV(raysFiltered);
                base.input6 = reflections.IsValid() ? ctx.SRV(reflections) : air::gpu::kInvalidIndex;
                base.input7 = adaptiveHeaders.IsValid() ? ctx.SRV(adaptiveHeaders) : air::gpu::kInvalidIndex;
                base.output = ctx.UAV(irradianceNow);
                base.input5 = ctx.UAV(specularNow);
                ctx.cmd.SetPipeline(gather);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            });

        const rhi::PipelineHandle temporal = mTemporal.Get(*mDevice);
        graph.AddPass(
            "ProbeGi.Temporal", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(depthRef, Access::SRVCompute);
                b.Read(normalRef, Access::SRVCompute);
                b.Read(motionRef, Access::SRVCompute);
                b.Read(irradianceNow, Access::SRVCompute);
                b.Read(specularNow, Access::SRVCompute);
                if (history)
                {
                    b.Read(irradiancePast, Access::SRVCompute);
                    b.Read(specularPast, Access::SRVCompute);
                    b.Read(surfacePast, Access::SRVCompute);
                }
                b.Write(irradianceOut, Access::UAVCompute);
                b.Write(specularOut, Access::UAVCompute);
                b.Write(surfaceOut, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.input0 = ctx.SRV(depthRef);
                base.input1 = ctx.SRV(normalRef);
                base.input2 = ctx.SRV(motionRef);
                base.input3 = ctx.SRV(irradianceNow);
                base.input4 = ctx.SRV(specularNow);
                base.input5 = history ? ctx.SRV(irradiancePast) : air::gpu::kInvalidIndex;
                base.input6 = history ? ctx.SRV(specularPast) : air::gpu::kInvalidIndex;
                base.input7 = history ? ctx.SRV(surfacePast) : air::gpu::kInvalidIndex;
                base.output = ctx.UAV(irradianceOut);
                base.rays = ctx.UAV(specularOut);
                base.counters = ctx.UAV(surfaceOut);
                ctx.cmd.SetPipeline(temporal);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            });

        // Under a denoiser of the surfaces' signals (nrd) the GI joins them, split by lobe as the path tracer's
        // indirect light would be, so that it is denoised with them (docs/architecture.md §8.14).
        const auto diffuseSignal = bb.FindTexture(slots::kDiffuseSignal);
        const auto specularSignal = bb.FindTexture(slots::kSpecularSignal);
        const bool signals = diffuseSignal && specularSignal && mView == gpu::kGiViewLit;
        const rdg::TextureRef diffuseSignalRef = signals ? *diffuseSignal : rdg::TextureRef{};
        const rdg::TextureRef specularSignalRef = signals ? *specularSignal : rdg::TextureRef{};
        const rhi::PipelineHandle composite = mComposite.Get(*mDevice);
        graph.AddPass(
            "ProbeGi.Composite", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(depthRef, Access::SRVCompute);
                b.Read(diffuseRef, Access::SRVCompute);
                b.Read(specularRef, Access::SRVCompute);
                b.Read(irradianceOut, Access::SRVCompute);
                b.Read(specularOut, Access::SRVCompute);
                b.Read(irradianceNow, Access::SRVCompute); // its a: where a probe fitted (view 8)
                if (emissionRef.IsValid())
                {
                    b.Read(emissionRef, Access::SRVCompute);
                }
                if (transmittanceRef.IsValid())
                {
                    b.Read(transmittanceRef, Access::SRVCompute);
                }
                b.Write(reflected, Access::UAVCompute);
                b.Write(sceneColor, Access::UAVCompute);
                if (signals)
                {
                    b.Write(diffuseSignalRef, Access::UAVCompute);
                    b.Write(specularSignalRef, Access::UAVCompute);
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.output = ctx.UAV(sceneColor);
                base.diffuseSignal = signals ? ctx.UAV(diffuseSignalRef) : air::gpu::kInvalidIndex;
                base.specularSignal = signals ? ctx.UAV(specularSignalRef) : air::gpu::kInvalidIndex;
                base.input0 = ctx.SRV(depthRef);
                base.input1 = ctx.SRV(diffuseRef);
                base.input2 = ctx.SRV(specularRef);
                base.input3 = ctx.SRV(irradianceOut);
                base.input4 = ctx.SRV(specularOut);
                base.input5 = emissionRef.IsValid() ? ctx.SRV(emissionRef) : air::gpu::kInvalidIndex;
                base.input6 = ctx.UAV(reflected);
                base.input7 = transmittanceRef.IsValid() ? ctx.SRV(transmittanceRef) : air::gpu::kInvalidIndex;
                base.rays = ctx.SRV(irradianceNow);
                ctx.cmd.SetPipeline(composite);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            });
        mHistoryValid = true;
        if (mView != mLastView)
        {
            // TAA would blend the frame and the view into one another; from then on it filters the view as it
            // does the frame (its edges, which the camera's jitter moves over, would flicker without it).
            context.blackboard.SetValue(slots::kHistoryReset, 1.0);
            mLastView = mView;
        }
    }

private:
    bool mEnabled = true;
    u32 mView = 0;
    VoxelSettings mVoxel;
    u32 mProbeTile = 16;
    u32 mFilterRadius = 0;
    u32 mRayFilterRadius = 1;
    bool mReflections = true;
    bool mAdaptive = true;
    u32 mAdaptiveRows = 0;
    bool mScreenTraces = true;
    u32 mScreenSteps = 16;
    float mScreenThickness = 0.03f;
    float mCacheDistance = 0.0f;
    float mTemporalBlend = 0.03f;
    float mStaticBlend = 0.01f;
    float mProbeBlend = 0.3f;
    float mDebugScale = 64.0f;
    bool mMultiBounce = true;

    rhi::Device* mDevice = nullptr;
    u32 mWidth = 0;
    u32 mHeight = 0;
    std::unique_ptr<VoxelScene> mVoxels;
    renderer::LazyComputePipeline mView3d;
    renderer::LazyComputePipeline mPlace;
    renderer::LazyComputePipeline mAdapt;
    renderer::LazyComputePipeline mTrace;
    renderer::LazyComputePipeline mRayFilter;
    renderer::LazyComputePipeline mSh;
    renderer::LazyComputePipeline mProbeTemporal;
    renderer::LazyComputePipeline mFilter;
    renderer::LazyComputePipeline mReflectionTrace;
    renderer::LazyComputePipeline mReflectionResolve;
    renderer::LazyComputePipeline mGather;
    renderer::LazyComputePipeline mTemporal;
    renderer::LazyComputePipeline mComposite;
    u32 mFrameIndex = 0;
    u32 mLastView = gpu::kGiViewLit; // the view TAA's history holds
    u32 mRelightsSeen = 0;
    u32 mFastHistoryFrames = 0; // frames left in which still pixels keep the moving pixels' shorter history
    std::optional<renderer::Camera> mPreviousCamera;
    hlsl::float2 mPreviousJitter{};
    u32 mProbesX = 0;
    u32 mProbesY = 0;
    bool mActive = false;
    bool mHistoryValid = false;
    rhi::TextureHandle mReflected;
    Access mReflectedAccess = Access::None;
    rhi::TextureHandle mIrradiance[2];
    rhi::TextureHandle mSpecular[2];
    rhi::TextureHandle mSurface[2];
    Access mIrradianceAccess[2] = {Access::None, Access::None};
    Access mSpecularAccess[2] = {Access::None, Access::None};
    Access mSurfaceAccess[2] = {Access::None, Access::None};
    u32 mCurrent = 0;
    // The screen probes and their accumulated harmonics, this frame's and last frame's.
    rhi::BufferHandle mProbes[2];
    rhi::BufferHandle mProbeSh[2];
    Access mProbesAccess[2] = {Access::None, Access::None};
    Access mProbeShAccess[2] = {Access::None, Access::None};
    u32 mProbeCount = 0;
    u32 mProbeCountTile = 0; // the probe tile they were made for
    u32 mProbeCurrent = 0;
    bool mProbeHistoryValid = false;

    void DestroyProbeHistory()
    {
        for (u32 i = 0; i < 2; ++i)
        {
            for (rhi::BufferHandle* b : {&mProbes[i], &mProbeSh[i]})
            {
                if (b->IsValid())
                {
                    mDevice->DestroyBuffer(*b);
                    *b = {};
                }
            }

            mProbesAccess[i] = Access::None;
            mProbeShAccess[i] = Access::None;
        }
        mProbeCount = 0;
    }
};

} // namespace
} // namespace air::pgi

namespace air::features
{

void Register_probe_gi(renderer::FeatureRegistry& registry)
{
    registry.Register({"probe_gi",
                       "Real-time GI without ray tracing: screen probes traced through a voxel distance field and a "
                       "world radiance cache",
                       [] { return std::make_unique<pgi::ProbeGiFeature>(); }});
}

} // namespace air::features
