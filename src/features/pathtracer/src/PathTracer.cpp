#include "air/features/pathtracer/PathTracer.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/features/pathtracer/Bsdf.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/Transfer.h"

#include "PathTracerShared.h"
#include "ReSTIR.h"
#include "ReSTIRShared.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace air::pt
{

using rhi::Access;

namespace
{

// Settings whose change invalidates the accumulated image.
bool SameImage(const PathTracerSettings& a, const PathTracerSettings& b)
{
    return a.maxBounces == b.maxBounces && a.accumulate == b.accumulate &&
           a.nextEventEstimation == b.nextEventEstimation && a.lightTreeBounces == b.lightTreeBounces &&
           a.restir == b.restir && a.furnace == b.furnace && a.debugView == b.debugView && a.debugScale == b.debugScale;
}

gpu::CameraData ToGpu(const renderer::Camera& camera, float aspect)
{
    gpu::CameraData out;
    out.position = {camera.position, std::tan(camera.verticalFov * 0.5f)};
    out.forward = {camera.Forward(), aspect};
    out.right = {camera.Right(), 0.0f};
    out.up = {camera.Up(), 0.0f};
    return out;
}

// Halton(2, 3) sub-pixel offsets in [-0.5, 0.5).
float Halton(u32 index, u32 base)
{
    float f = 1.0f;
    float r = 0.0f;
    for (u32 i = index; i > 0; i /= base)
    {
        f /= float(base);
        r += f * float(i % base);
    }
    return r;
}

bool IsDisplayDebugView(u32 view)
{
    return view != gpu::kDebugNone && view != gpu::kDebugDirect && view != gpu::kDebugIndirect;
}

} // namespace

PathTracer::PathTracer(rhi::Device& device, const PathTracerSettings& settings)
    : mDevice(device), mSettings(settings), mLastSettings(settings)
{
}

PathTracer::~PathTracer()
{
    for (rhi::TextureHandle texture : {mAccumulation})
    {
        if (texture.IsValid())
        {
            mDevice.DestroyTexture(texture);
        }
    }
    for (rhi::BufferHandle buffer : {mBsdfLut, mFrameRing})
    {
        if (buffer.IsValid())
        {
            mDevice.DestroyBuffer(buffer);
        }
    }
    for (rhi::PipelineHandle pipeline : {mTracePipeline, mTracePipelineSer, mTracePipelineGi, mResolvePipeline})
    {
        if (pipeline.IsValid())
        {
            mDevice.DestroyPipeline(pipeline);
        }
    }
}

bool PathTracer::Initialize(const std::filesystem::path& shaderDir)
{
    if (!mDevice.GetCapabilities().rayTracing)
    {
        AIR_LOG(Error, "PathTracer", "device '{}' does not support hardware ray tracing",
                mDevice.GetCapabilities().adapterName);
        return false;
    }
    rhi::RayTracingPipelineDesc trace;
    trace.library = renderer::LoadShader(mDevice, shaderDir, "PathTracer.lib", "");
    trace.rayGen = "RayGen";
    trace.missShaders = {"Miss"};
    trace.hitGroups = {{"HitGroup", "ClosestHit", ""}};
    trace.maxPayloadBytes = 20;
    trace.maxRecursionDepth = 1;
    trace.pushConstantBytes = sizeof(gpu::TraceConstants);
    trace.debugName = "PathTracer.Trace";
    mTracePipeline = mDevice.CreateRayTracingPipeline(trace);
    if (mDevice.GetCapabilities().shaderExecutionReordering)
    {
        // Same entry points, built from the same kernel with AIR_PT_SER.
        trace.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerSer.lib", "");
        trace.debugName = "PathTracer.TraceSer";
        if (!trace.library.bytecode.empty())
        {
            mTracePipelineSer = mDevice.CreateRayTracingPipeline(trace);
        }
        AIR_LOG(Info, "PathTracer", "shader execution reordering {}",
                mTracePipelineSer.IsValid() ? "available" : "unavailable (PathTracerSer.lib missing)");
    }
    // The kernel that records ReSTIR GI's candidates (AIR_PT_RESTIR_GI).
    trace.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerGi.lib", "");
    trace.debugName = "PathTracer.TraceGi";
    mTracePipelineGi = mDevice.CreateRayTracingPipeline(trace);

    rhi::ComputePipelineDesc resolve;
    resolve.shader = renderer::LoadShader(mDevice, shaderDir, "Resolve.cs", "Main");
    resolve.pushConstantBytes = sizeof(gpu::ResolveConstants);
    resolve.debugName = "PathTracer.Resolve";
    mResolvePipeline = mDevice.CreateComputePipeline(resolve);

    mAccumulation =
        mDevice.CreateTexture({.format = rhi::Format::RGBA32_FLOAT,
                               .width = mSettings.width,
                               .height = mSettings.height,
                               .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                               .debugName = "PathTracer.Accumulation"});
    mAccumulationAccess = Access::None;

    const std::span<const float> lut = GetBsdfLut();
    mBsdfLut = renderer::CreateBufferWithData(
        mDevice,
        {.stride = sizeof(float), .usage = rhi::BufferUsage::ShaderResource, .debugName = "PathTracer.BsdfLut"},
        lut.data(), lut.size_bytes());

    // Optional: without their shaders the path tracer estimates all the light itself.
    mReSTIR = std::make_unique<ReSTIR>(mDevice);
    mReSTIR->Initialize(shaderDir);

    mFrameRingSize = mDevice.GetFramesInFlight() + 1;
    mFrameRing = mDevice.CreateBuffer({.size = mFrameRingSize * sizeof(gpu::FrameData),
                                       .stride = sizeof(gpu::FrameData),
                                       .usage = rhi::BufferUsage::ShaderResource,
                                       .memory = rhi::MemoryLocation::DeviceLocalHost,
                                       .debugName = "PathTracer.FrameData"});

    return mTracePipeline.IsValid() && mResolvePipeline.IsValid() && mAccumulation.IsValid() && mBsdfLut.IsValid() &&
           mFrameRing.IsValid();
}

void PathTracer::ResetAccumulation()
{
    mResetPending = true;
    mAccumulatedSamples = 0;
}

void PathTracer::AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const renderer::Camera& camera,
                           renderer::Blackboard& blackboard, rdg::TextureRef output)
{
    AIR_CHECK(mAccumulation.IsValid(), "PathTracer::AddPasses before Initialize()");
    namespace slots = renderer::slots;

    const bool cameraChanged = !mLastCamera || !(*mLastCamera == camera);
    const bool historyInvalid =
        !mLastCamera || scene.version != mLastSceneVersion || !SameImage(mSettings, mLastSettings);
    if (cameraChanged || historyInvalid || !mSettings.accumulate)
    {
        ResetAccumulation();
    }
    const renderer::Camera previousCamera = mLastCamera.value_or(camera);
    mLastCamera = camera;
    mLastSceneVersion = scene.version;
    mLastSettings = mSettings;
    blackboard.SetValue(slots::kHistoryReset, historyInvalid ? 1.0 : 0.0);
    blackboard.SetValue(slots::kOutputIsDisplay, IsDisplayDebugView(mSettings.debugView) ? 1.0 : 0.0);

    // Converged: keep showing the result without tracing more samples.
    const bool trace = !IsConverged();
    const u32 spp = trace ? std::min(mSettings.samplesPerPixelPerFrame,
                                     mSettings.accumulate ? mSettings.targetSamples - mAccumulatedSamples : ~0u)
                          : 0u;

    // ReSTIR replaces the path's own estimate of the direct (DI) and indirect (GI) light at the first
    // surface, for the lit and direct / indirect views.
    const u32 view = mSettings.debugView;
    const bool lightView = view == gpu::kDebugNone || view == gpu::kDebugDirect || view == gpu::kDebugIndirect;
    auto on = [&](u32 mode) { return mode == kReSTIRAlways || (mode == kReSTIRRealTime && !mSettings.accumulate); };
    const bool diWanted = on(mSettings.restir.di) && mSettings.nextEventEstimation && !mSettings.furnace &&
                          scene.lightCount > 0 && lightView && mReSTIR->IsDiAvailable();
    const bool giWanted =
        on(mSettings.restir.gi) && lightView && mReSTIR->IsGiAvailable() && mTracePipelineGi.IsValid();
    const bool restirDi = diWanted && spp > 0;
    const bool restirGi = giWanted && spp > 0;
    AIR_CHECK(!restirDi || scene.lightCount <= gpu::kReSTIRMaxLight,
              "ReSTIR DI: {} lights exceed the reservoirs' 24 bits", scene.lightCount);
    if (!diWanted && !giWanted)
    {
        mReSTIR->Invalidate();
    }
    if (spp > 0)
    {
        mUsedReSTIRDI = restirDi;
        mUsedReSTIRGI = restirGi;
    }

    const float aspect = float(mSettings.width) / float(mSettings.height);
    gpu::FrameData frame{};
    frame.camera = ToGpu(camera, aspect);
    frame.previousCamera = ToGpu(previousCamera, aspect);
    float exposure = renderer::ExposureScale(camera.ev100);
    frame.sky = scene.sky;
    if (mSettings.furnace)
    {
        // A uniform sky (white for a physical one) mapped to middle grey and no sun: a correct
        // furnace image is a flat 0.18.
        const hlsl::float3 uniform = scene.sky.width == 0 ? scene.sky.uniformRadiance : hlsl::float3(1.0f);
        const float luminance = std::max(hlsl::dot(uniform, hlsl::float3(0.2126f, 0.7152f, 0.0722f)), 0.0f);
        frame.sky.width = 0;
        frame.sky.height = 0;
        frame.sky.uniformRadiance = luminance > 0.0f ? uniform : hlsl::float3(1.0f);
        frame.sky.sunRadiance = hlsl::float3(0.0f);
        exposure = luminance > 0.0f ? 0.18f / luminance : 0.18f;
    }
    frame.exposure = exposure;
    if (!mSettings.accumulate)
    {
        const u32 phase = u32(mFrameIndex % std::max(mSettings.jitterPhases, 1u)) + 1;
        frame.jitter = {Halton(phase, 2) - 0.5f, Halton(phase, 3) - 0.5f};
    }
    blackboard.SetValue(slots::kJitterX, frame.jitter.x);
    blackboard.SetValue(slots::kJitterY, frame.jitter.y);
    frame.frameIndex = mFrameIndex++;
    frame.flags = (mResetPending ? gpu::kFlagReset : 0u) | (mSettings.furnace ? gpu::kFlagFurnace : 0u) |
                  (mSettings.accumulate ? 0u : gpu::kFlagGlobalJitter) | (mSettings.guides ? gpu::kFlagGuides : 0u) |
                  (mSettings.nextEventEstimation ? 0u : gpu::kFlagNoNee) | (restirDi ? gpu::kFlagReSTIR : 0u) |
                  (restirGi ? gpu::kFlagReSTIRGI : 0u);
    frame.lightTreeBounces = mSettings.lightTreeBounces;
    frame.width = mSettings.width;
    frame.height = mSettings.height;
    frame.samplesPerPixel = spp;
    frame.maxBounces = mSettings.maxBounces;
    frame.lightCount = scene.lightCount;
    frame.debugView = mSettings.debugView;
    frame.debugScale = std::max(mSettings.debugScale, 1e-6f);

    const u32 slot = frame.frameIndex % mFrameRingSize;
    auto* ring = static_cast<gpu::FrameData*>(mDevice.GetMappedPointer(mFrameRing));
    AIR_CHECK(ring, "PathTracer.FrameData is not host visible");
    std::memcpy(&ring[slot], &frame, sizeof(frame));

    // The accumulation buffer persists across frames: import it and remember the state it ends in.
    const Access endAccess = Access::SRVCompute;
    rdg::TextureRef accumulation = graph.ImportTexture(mAccumulation, mAccumulationAccess, endAccess);
    mAccumulationAccess = endAccess;

    const u32 width = mSettings.width;
    const u32 height = mSettings.height;
    gpu::TraceConstants constants{};
    constants.frameData = mDevice.GetBindlessSRV(mFrameRing);
    constants.frameSlot = slot;
    constants.tlas = mDevice.GetBindlessSRV(scene.tlas);
    constants.instances = mDevice.GetBindlessSRV(scene.instances);
    constants.materials = mDevice.GetBindlessSRV(scene.materials);
    constants.lights = mDevice.GetBindlessSRV(scene.lights);
    constants.instanceLights = mDevice.GetBindlessSRV(scene.instanceLights);
    constants.lightTree = mDevice.GetBindlessSRV(scene.lightTree);
    constants.bsdfLut = mDevice.GetBindlessSRV(mBsdfLut);
    constants.skyTable = mDevice.GetBindlessSRV(scene.skyTable);

    if (spp > 0)
    {
        mResetPending = false;
        mAccumulatedSamples += spp;

        struct Guides
        {
            rdg::TextureRef diffuseAlbedo, specularAlbedo, normalRoughness, viewDepth, motionVectors;
        } guides;
        if (mSettings.guides)
        {
            auto make = [&](rhi::Format format, const char* name)
            {
                return graph.CreateTexture(
                    {.format = format,
                     .width = width,
                     .height = height,
                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                     .debugName = name});
            };
            guides.diffuseAlbedo = make(rhi::Format::RGBA16_FLOAT, "PathTracer.DiffuseAlbedo");
            guides.specularAlbedo = make(rhi::Format::RGBA16_FLOAT, "PathTracer.SpecularAlbedo");
            guides.normalRoughness = make(rhi::Format::RGBA16_FLOAT, "PathTracer.NormalRoughness");
            guides.viewDepth = make(rhi::Format::R32_FLOAT, "PathTracer.ViewDepth");
            guides.motionVectors = make(rhi::Format::RG16_FLOAT, "PathTracer.MotionVectors");
            blackboard.SetTexture(slots::kDiffuseAlbedo, guides.diffuseAlbedo);
            blackboard.SetTexture(slots::kSpecularAlbedo, guides.specularAlbedo);
            blackboard.SetTexture(slots::kNormalRoughness, guides.normalRoughness);
            blackboard.SetTexture(slots::kViewDepth, guides.viewDepth);
            blackboard.SetTexture(slots::kMotionVectors, guides.motionVectors);
        }

        ReSTIR::TraceOutputs restirOutputs;
        if (restirDi || restirGi)
        {
            restirOutputs = mReSTIR->BeginFrame(graph, width, height, !historyInvalid, restirDi, restirGi);
        }
        const rdg::BufferRef primaryHits = restirOutputs.primaryHits;
        const rdg::BufferRef giReservoirs = restirOutputs.giReservoirs;

        const rhi::PipelineHandle pipeline =
            restirGi ? mTracePipelineGi : (UsesShaderExecutionReordering() ? mTracePipelineSer : mTracePipeline);
        graph.AddPass(
            "PathTracer.Trace", rdg::PassFlags::RayTracing,
            [&](rdg::PassBuilder& b)
            {
                b.Write(accumulation, Access::UAVCompute);
                for (rdg::BufferRef buffer : {primaryHits, giReservoirs})
                {
                    if (buffer.IsValid())
                    {
                        b.Write(buffer, Access::UAVCompute);
                    }
                }
                for (rdg::TextureRef t : {guides.diffuseAlbedo, guides.specularAlbedo, guides.normalRoughness,
                                          guides.viewDepth, guides.motionVectors})
                {
                    if (t.IsValid())
                    {
                        b.Write(t, Access::UAVCompute);
                    }
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                constants.accumulation = ctx.UAV(accumulation);
                if (guides.diffuseAlbedo.IsValid())
                {
                    constants.diffuseAlbedo = ctx.UAV(guides.diffuseAlbedo);
                    constants.specularAlbedo = ctx.UAV(guides.specularAlbedo);
                    constants.normalRoughness = ctx.UAV(guides.normalRoughness);
                    constants.viewDepth = ctx.UAV(guides.viewDepth);
                    constants.motionVectors = ctx.UAV(guides.motionVectors);
                }
                if (primaryHits.IsValid())
                {
                    constants.primaryHits = ctx.UAV(primaryHits);
                }
                if (giReservoirs.IsValid())
                {
                    constants.giReservoirs = ctx.UAV(giReservoirs);
                }
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&constants, sizeof(constants));
                ctx.cmd.DispatchRays(width, height, 1);
            });

        if (restirDi || restirGi)
        {
            gpu::ReSTIRConstants restirScene{};
            restirScene.frameData = constants.frameData;
            restirScene.frameSlot = constants.frameSlot;
            restirScene.tlas = constants.tlas;
            restirScene.instances = constants.instances;
            restirScene.materials = constants.materials;
            restirScene.lights = constants.lights;
            restirScene.instanceLights = constants.instanceLights;
            restirScene.lightTree = constants.lightTree;
            restirScene.bsdfLut = constants.bsdfLut;
            restirScene.skyTable = constants.skyTable;
            mReSTIR->AddPasses(graph, mSettings.restir, restirScene, accumulation, view != gpu::kDebugIndirect,
                               view != gpu::kDebugDirect);
        }
    }

    const rhi::PipelineHandle resolvePipeline = mResolvePipeline;
    const u32 frameData = constants.frameData;
    graph.AddPass(
        "PathTracer.Resolve", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(accumulation, Access::SRVCompute);
            b.Write(output, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx)
        {
            gpu::ResolveConstants resolve{};
            resolve.frameData = frameData;
            resolve.frameSlot = slot;
            resolve.accumulation = ctx.SRV(accumulation);
            resolve.output = ctx.UAV(output);
            ctx.cmd.SetPipeline(resolvePipeline);
            ctx.cmd.PushConstants(&resolve, sizeof(resolve));
            ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        });
}

} // namespace air::pt
