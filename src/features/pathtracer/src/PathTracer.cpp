#include "air/features/pathtracer/PathTracer.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/features/pathtracer/Bsdf.h"
#include "air/features/pathtracer/NeuralCoat.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/Transfer.h"

#include "CloudMarcher.h"
#include "PathTracerShared.h"
#include "RadianceCache.h"
#include "ReSTIR.h"
#include "ReSTIRShared.h"
#include "VolumeMarcher.h"

#include <algorithm>
#include <chrono>
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
    // The wind's step only sets how far the clouds drift between real-time frames (an app may follow
    // its frame time with it), and real-time frames advance their clock themselves.
    CloudMarchSettings cloudMarch = a.cloudMarch;
    cloudMarch.windStep = b.cloudMarch.windStep;
    if (!a.accumulate && !b.accumulate)
    {
        cloudMarch.time = b.cloudMarch.time;
    }
    return a.maxBounces == b.maxBounces && a.accumulate == b.accumulate &&
           a.nextEventEstimation == b.nextEventEstimation && a.lightTreeBounces == b.lightTreeBounces &&
           a.cloudEvents == b.cloudEvents && a.volumeEvents == b.volumeEvents && a.restir == b.restir &&
           a.nrc == b.nrc && a.volumeMarch == b.volumeMarch && cloudMarch == b.cloudMarch && a.furnace == b.furnace &&
           a.debugView == b.debugView && a.debugScale == b.debugScale && a.coatModel == b.coatModel;
}

gpu::CameraData ToGpu(const renderer::Camera& camera, float aspect)
{
    gpu::CameraData out{};
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
    return view != gpu::kDebugNone && view != gpu::kDebugDirect && view != gpu::kDebugIndirect &&
           view != gpu::kDebugNrcCache;
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
    for (rhi::BufferHandle buffer : {mBsdfLut, mFrameRing, mNeuralCoat})
    {
        if (buffer.IsValid())
        {
            mDevice.DestroyBuffer(buffer);
        }
    }
    mTraceNrc.Destroy(mDevice);
    mTraceNrcPortable.Destroy(mDevice);
    for (rhi::PipelineHandle pipeline :
         {mTracePipeline, mTracePipelineSer, mTracePipelineGi, mTracePipelineVolume, mTracePipelineVolumeShadows,
          mTracePipelineNeural, mTracePipelineGiNeural, mResolvePipeline})
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
    // The plain kernel with the neural parts (AIR_PT_NEURAL), created the first time a scene needs them.
    mTraceNeuralDesc = trace;
    mTraceNeuralDesc.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerNeural.lib", "");
    mTraceNeuralDesc.debugName = "PathTracer.TraceNeural";
    // The kernel that records ReSTIR GI's candidates (AIR_PT_RESTIR_GI), and its twin with AIR_PT_NEURAL.
    trace.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerGi.lib", "");
    trace.debugName = "PathTracer.TraceGi";
    mTracePipelineGi = mDevice.CreateRayTracingPipeline(trace);
    mTraceGiNeuralDesc = trace;
    mTraceGiNeuralDesc.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerGiNeural.lib", "");
    mTraceGiNeuralDesc.debugName = "PathTracer.TraceGiNeural";
    // The kernel that tracks the scene's volume (AIR_PT_VOLUME), created the first time a scene has one.
    mTraceVolumeDesc = trace;
    mTraceVolumeDesc.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerVolume.lib", "");
    mTraceVolumeDesc.debugName = "PathTracer.TraceVolume";
    // The kernel under real-time volumes (AIR_PT_VOLUME_SHADOWS), likewise.
    mTraceVolumeShadowsDesc = trace;
    mTraceVolumeShadowsDesc.library = renderer::LoadShader(mDevice, shaderDir, "PathTracerVolumeShadows.lib", "");
    mTraceVolumeShadowsDesc.debugName = "PathTracer.TraceVolumeShadows";
    // The kernels whose paths end in the neural radiance cache, compute shaders with inline ray
    // queries: on the matrix units (Shader Model 6.10, built with the preview compiler) and in float32.
    if (mDevice.GetCapabilities().linearAlgebra)
    {
        mTraceNrc.Load(mDevice, shaderDir, "PathTracerNrc.cs", "PathTracer.TraceNrc", sizeof(gpu::TraceConstants));
    }
    mTraceNrcPortable.Load(mDevice, shaderDir, "PathTracerNrcPortable.cs", "PathTracer.TraceNrcPortable",
                           sizeof(gpu::TraceConstants));

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
    mRadianceCache = std::make_unique<RadianceCache>(mDevice);
    mRadianceCache->Initialize(shaderDir);
    mVolumeMarcher = std::make_unique<VolumeMarcher>(mDevice);
    mVolumeMarcher->Initialize(shaderDir);
    mCloudMarcher = std::make_unique<CloudMarcher>(mDevice);
    mCloudMarcher->Initialize(shaderDir);

    mFrameRingSize = mDevice.GetFramesInFlight() + 1;
    mFrameRing = mDevice.CreateBuffer({.size = mFrameRingSize * sizeof(gpu::FrameData),
                                       .stride = sizeof(gpu::FrameData),
                                       .usage = rhi::BufferUsage::ShaderResource,
                                       .memory = rhi::MemoryLocation::DeviceLocalHost,
                                       .debugName = "PathTracer.FrameData"});

    return mTracePipeline.IsValid() && mResolvePipeline.IsValid() && mAccumulation.IsValid() && mBsdfLut.IsValid() &&
           mFrameRing.IsValid();
}

bool PathTracer::RadianceCacheUsesLinearAlgebra() const
{
    return mUsedNrc && mRadianceCache->UsesLinearAlgebra();
}

u32 PathTracer::GetRadianceCacheSteps() const
{
    return mRadianceCache ? mRadianceCache->GetStep() : 0;
}

u32 PathTracer::GetVolumeMarchLights() const
{
    return mVolumeMarcher ? mVolumeMarcher->GetLightCount() : 0u;
}

void PathTracer::UpdateNeuralCoat(const renderer::GpuScene& scene)
{
    const std::vector<CoatMaterialSample> materials =
        scene.desc ? CollectCoatMaterials(*scene.desc) : std::vector<CoatMaterialSample>{};
    u64 key = 0xcbf29ce484222325ull;
    const auto* bytes = reinterpret_cast<const u8*>(materials.data());
    for (size_t i = 0; i < materials.size() * sizeof(CoatMaterialSample); ++i)
    {
        key = (key ^ bytes[i]) * 0x100000001b3ull;
    }
    if (key == mNeuralCoatKey && (materials.empty() || mNeuralCoat.IsValid()))
    {
        return;
    }
    mNeuralCoatKey = key;
    if (mNeuralCoat.IsValid())
    {
        mDevice.WaitIdle(); // frames in flight may still read the old network
        mDevice.DestroyBuffer(mNeuralCoat);
        mNeuralCoat = {};
    }
    mNeuralCoatLoss = 0.0f;
    if (materials.empty())
    {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const NeuralCoatResult coat = TrainNeuralCoat(materials, {}, mTasks);
    mNeuralCoatLoss = coat.loss;
    std::vector<float> padded = coat.parameters; // a constant buffer: 256-byte multiples
    padded.resize((padded.size() + 63) / 64 * 64, 0.0f);
    static_assert(gpu::kNeuralCoatParameterVectors * 4 == 980, "PathTracerShared.h: kNeuralCoatParameters / 4");
    mNeuralCoat = renderer::CreateBufferWithData(
        mDevice, {.usage = rhi::BufferUsage::Constant, .debugName = "PathTracer.NeuralCoat"}, padded.data(),
        padded.size() * sizeof(float));
    AIR_LOG(Info, "PathTracer", "neural coat for {} coated material samples: loss {:.4f}, {} ({:.1f} s)",
            materials.size(), coat.loss, coat.cached ? "cached" : "trained",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
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
    // The neural coat follows the coated materials: at once when there is none yet, else once the scene
    // has stopped changing for a while (an editor dragging a slider keeps the last network until then).
    if (scene.version != mLastSceneVersion)
    {
        mNeuralCoatPending = true;
        mNeuralCoatStableFrames = 0;
    }
    else
    {
        ++mNeuralCoatStableFrames;
    }
    if (mNeuralCoatPending && (!mNeuralCoat.IsValid() || mNeuralCoatStableFrames >= kNeuralCoatSettleFrames))
    {
        mNeuralCoatPending = false;
        UpdateNeuralCoat(scene);
    }
    mLastSceneVersion = scene.version;
    mLastSettings = mSettings;
    blackboard.SetValue(slots::kHistoryReset, historyInvalid ? 1.0 : 0.0);
    blackboard.SetValue(slots::kOutputIsDisplay, IsDisplayDebugView(mSettings.debugView) ? 1.0 : 0.0);
    blackboard.SetValue(slots::kRealTime, mSettings.accumulate ? 0.0 : 1.0);

    // Converged: keep showing the result without tracing more samples.
    const bool trace = !IsConverged();
    const u32 spp = trace ? std::min(mSettings.samplesPerPixelPerFrame,
                                     mSettings.accumulate ? mSettings.targetSamples - mAccumulatedSamples : ~0u)
                          : 0u;

    // ReSTIR replaces the path's own estimate of the direct (DI) and indirect (GI) light at the first
    // surface, for the lit and direct / indirect views.
    const u32 view = mSettings.debugView;
    const bool lightView = view == gpu::kDebugNone || view == gpu::kDebugDirect || view == gpu::kDebugIndirect;
    // Volumes in the scene: ray marched over the surfaces (real-time volumes), or tracked by a kernel of
    // their own. Either way paths do all the work (ReSTIR's first surface and the cache's queries do not
    // know about media).
    const bool volume = !scene.volumes.empty();
    auto on = [&](u32 mode) { return mode == kReSTIRAlways || (mode == kReSTIRRealTime && !mSettings.accumulate); };
    const bool marchWanted = volume && on(mSettings.volumeMarch.mode) && mVolumeMarcher->IsAvailable();
    if (marchWanted && !mTracePipelineVolumeShadows.IsValid())
    {
        mTracePipelineVolumeShadows = mDevice.CreateRayTracingPipeline(mTraceVolumeShadowsDesc);
        AIR_CHECK(mTracePipelineVolumeShadows.IsValid(), "PathTracer: the real-time volumes' kernel did not compile");
    }
    const bool march = marchWanted && mTracePipelineVolumeShadows.IsValid();
    if (volume && !march && !mTracePipelineVolume.IsValid() && !mTraceVolumeDesc.library.bytecode.empty())
    {
        mTracePipelineVolume = mDevice.CreateRayTracingPipeline(mTraceVolumeDesc);
        AIR_CHECK(mTracePipelineVolume.IsValid(), "PathTracer: the volume kernel did not compile");
    }
    // Neural texture sets decoded on sample and coated materials' coat models: by the plain and ReSTIR GI
    // kernels' twins (and ReSTIR's); the others read the sets' coarse copies and the coats'
    // approximation.
    const bool neural = (scene.neuralTextures || scene.coatedMaterials) && !volume;
    auto create = [&](rhi::PipelineHandle& pipeline, const rhi::RayTracingPipelineDesc& desc)
    {
        if (neural && !pipeline.IsValid())
        {
            pipeline = mDevice.CreateRayTracingPipeline(desc);
            AIR_CHECK(pipeline.IsValid(), "PathTracer: {} did not compile", desc.debugName);
        }
    };
    create(mTracePipelineNeural, mTraceNeuralDesc);
    create(mTracePipelineGiNeural, mTraceGiNeuralDesc);
    const bool diWanted = on(mSettings.restir.di) && mSettings.nextEventEstimation && !mSettings.furnace &&
                          scene.lightCount > 0 && lightView && !volume && mReSTIR->IsDiAvailable();
    // The neural radiance cache: its own kernel variant, which leaves the indirect light to the cache
    // (no ReSTIR GI). On the matrix units when asked for and available, else in float32.
    const bool nrcView = lightView || view == gpu::kDebugNrcCache;
    const bool nrcOn =
        mSettings.nrc.mode == kNrcAlways || (mSettings.nrc.mode == kNrcRealTime && !mSettings.accumulate);
    const bool nrcLinearAlgebra =
        mSettings.nrc.linearAlgebra && mTraceNrc.IsAvailable() && mRadianceCache->IsAvailable(true);
    // The trace kernel is created the first time the cache runs.
    const rhi::PipelineHandle nrcPipeline = !nrcOn || !nrcView || volume         ? rhi::PipelineHandle{}
                                            : nrcLinearAlgebra                   ? mTraceNrc.Get(mDevice)
                                            : mRadianceCache->IsAvailable(false) ? mTraceNrcPortable.Get(mDevice)
                                                                                 : rhi::PipelineHandle{};
    const bool nrcWanted = nrcPipeline.IsValid();
    const bool giWanted = on(mSettings.restir.gi) && lightView && !nrcWanted && !volume && mReSTIR->IsGiAvailable() &&
                          mTracePipelineGi.IsValid();
    const bool restirDi = diWanted && spp > 0;
    const bool restirGi = giWanted && spp > 0;
    const bool nrc = nrcWanted && spp > 0;
    AIR_CHECK(!restirDi || scene.lightCount <= gpu::kReSTIRMaxLight,
              "ReSTIR DI: {} lights exceed the reservoirs' 24 bits", scene.lightCount);
    if (!diWanted && !giWanted)
    {
        mReSTIR->Invalidate();
    }
    // The real-time clouds: their maps instead of walks through them (not in the furnace, where the
    // walks check that the clouds conserve energy, nor with real-time volumes, whose shadow maps do not
    // know the clouds).
    const bool cloudMarch = scene.clouds.enabled != 0 && on(mSettings.cloudMarch.mode) && !mSettings.furnace &&
                            !march && mCloudMarcher->IsAvailable();
    if (spp > 0)
    {
        mUsedCloudMarch = cloudMarch;
        mUsedReSTIRDI = restirDi;
        mUsedReSTIRGI = restirGi;
        mUsedNrc = nrc;
        mUsedVolumeMarch = march;
        mUsedNeuralKernels =
            neural && !nrc && (restirGi ? mTracePipelineGiNeural.IsValid() : mTracePipelineNeural.IsValid());
    }

    const float aspect = float(mSettings.width) / float(mSettings.height);
    gpu::FrameData frame{};
    frame.camera = ToGpu(camera, aspect);
    frame.previousCamera = ToGpu(previousCamera, aspect);
    float exposure = renderer::ExposureScale(camera.ev100);
    frame.sky = scene.sky;
    if (scene.aerialInscatter.IsValid())
    {
        frame.sky.aerialInscatter = mDevice.GetBindlessSRV(scene.aerialInscatter);
        frame.sky.aerialTransmittance = mDevice.GetBindlessSRV(scene.aerialTransmittance);
    }
    else
    {
        frame.sky.aerialDistance = 0.0f;
    }
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
        frame.sky.aerialDistance = 0.0f;
        exposure = luminance > 0.0f ? 0.18f / luminance : 0.18f;
    }
    frame.exposure = exposure;
    frame.clouds = scene.clouds;
    if (scene.clouds.enabled != 0)
    {
        frame.clouds.baseTexture = mDevice.GetBindlessSRV(scene.cloudBase);
        frame.clouds.detailTexture = mDevice.GetBindlessSRV(scene.cloudDetail);
        frame.clouds.majorants = mDevice.GetBindlessSRV(scene.cloudMajorants);
        // The wind: real-time frames advance the clouds' clock, accumulation holds it. The detail
        // drifts half again as fast as the base and rises, so the fringes billow (Nubis). Offsets wrap
        // to their periods (the tile holds a whole number of detail periods).
        if (!mSettings.accumulate && spp > 0)
        {
            mSettings.cloudMarch.time += mSettings.cloudMarch.windStep;
        }
        const double time = double(mSettings.cloudMarch.time);
        auto wrap = [](double x, float period) { return float(x - double(period) * std::round(x / double(period))); };
        const float speed = std::hypot(frame.clouds.windX, frame.clouds.windZ);
        frame.clouds.baseOffsetX = wrap(double(frame.clouds.windX) * time, frame.clouds.tileSize);
        frame.clouds.baseOffsetZ = wrap(double(frame.clouds.windZ) * time, frame.clouds.tileSize);
        frame.clouds.detailOffsetX = wrap(1.5 * double(frame.clouds.windX) * time, frame.clouds.detailPeriod);
        frame.clouds.detailOffsetY = wrap(0.25 * double(speed) * time, frame.clouds.detailPeriod);
        frame.clouds.detailOffsetZ = wrap(1.5 * double(frame.clouds.windZ) * time, frame.clouds.detailPeriod);
        // The cirrus is not periodic: its offsets wrap every 1000 km (once in 11 hours at 24 m/s).
        frame.clouds.cirrusOffsetX = wrap(2.0 * double(frame.clouds.windX) * time, 1.0e6f);
        frame.clouds.cirrusOffsetZ = wrap(2.0 * double(frame.clouds.windZ) * time, 1.0e6f);
    }
    frame.cloudEvents = mSettings.cloudEvents;
    frame.volumeEvents = mSettings.volumeEvents;
    frame.volumeCount = u32(scene.volumes.size());
    frame.volumeTable = volume ? mDevice.GetBindlessSRV(scene.volumeTable) : 0u;
    frame.textureSets = mDevice.GetBindlessSRV(scene.textureSetTable);
    // Bsdf.hlsli kCoat* (the setting + 1); without a network the neural coat is the approximation.
    frame.coatModel = mSettings.coatModel == kCoatModelNeural && !mNeuralCoat.IsValid()
                          ? kCoatModelApproximate + 1
                          : std::min(mSettings.coatModel, kCoatModelApproximate) + 1;
    frame.neuralCoat = mNeuralCoat.IsValid() ? mDevice.GetBindlessCBV(mNeuralCoat) : 0u;
    if (!mSettings.accumulate)
    {
        const u32 phase = u32(mFrameIndex % std::max(mSettings.jitterPhases, 1u)) + 1;
        frame.jitter = {Halton(phase, 2) - 0.5f, Halton(phase, 3) - 0.5f};
    }
    else if (march)
    {
        // The march follows the trace's camera rays: all of a frame's samples take the frame's jitter,
        // which runs through the Halton sequence as samples accumulate.
        frame.jitter = {Halton(mAccumulatedSamples + 1, 2) - 0.5f, Halton(mAccumulatedSamples + 1, 3) - 0.5f};
    }
    blackboard.SetValue(slots::kJitterX, frame.jitter.x);
    blackboard.SetValue(slots::kJitterY, frame.jitter.y);
    frame.frameIndex = mFrameIndex++;
    // The first surface's light split by lobe, for a denoiser of the surfaces (real time, lit view).
    const bool signals = mSettings.signals && mSettings.guides && !mSettings.accumulate && view == gpu::kDebugNone;
    frame.flags = (mResetPending ? gpu::kFlagReset : 0u) | (mSettings.furnace ? gpu::kFlagFurnace : 0u) |
                  (mSettings.accumulate ? 0u : gpu::kFlagGlobalJitter) | (mSettings.guides ? gpu::kFlagGuides : 0u) |
                  (mSettings.nextEventEstimation ? 0u : gpu::kFlagNoNee) | (restirDi ? gpu::kFlagReSTIR : 0u) |
                  (restirGi ? gpu::kFlagReSTIRGI : 0u) | (nrc ? gpu::kFlagNrc : 0u) |
                  (march ? gpu::kFlagVolumeMarch | gpu::kFlagGlobalJitter : 0u) | (signals ? gpu::kFlagSignals : 0u) |
                  (cloudMarch ? gpu::kFlagCloudMaps : 0u);
    frame.lightTreeBounces = mSettings.lightTreeBounces;
    frame.width = mSettings.width;
    frame.height = mSettings.height;
    frame.samplesPerPixel = spp;
    frame.maxBounces = mSettings.maxBounces;
    frame.lightCount = scene.lightCount;
    frame.debugView = mSettings.debugView;
    frame.debugScale = std::max(mSettings.debugScale, 1e-6f);
    if (nrc)
    {
        mRadianceCache->BeginFrame(graph, scene, mSettings.nrc, nrcLinearAlgebra, mSettings.width, mSettings.height,
                                   exposure, frame.frameIndex, frame.nrc);
    }
    if (march && spp > 0)
    {
        mVolumeMarcher->BeginFrame(graph, scene, mSettings.volumeMarch, frame.volumeShadows);
    }
    frame.cloudLayer = air::gpu::kInvalidIndex;
    frame.cloudDepth = air::gpu::kInvalidIndex;
    if (cloudMarch && spp > 0)
    {
        mCloudMarcher->BeginFrame(scene, mSettings.width, mSettings.height, !historyInvalid, frame);
    }

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

    // Under real-time volumes with a denoiser: the trace's samples and the frame resolved without the
    // volumes (slots::kColorBeforeTransparency).
    rdg::TextureRef frameSamples;
    rdg::TextureRef colorBefore;
    if (spp > 0)
    {
        mResetPending = false;
        mAccumulatedSamples += spp;

        DenoiserGuides guides;
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
        rdg::TextureRef diffuseSignal;
        rdg::TextureRef specularSignal;
        if (signals)
        {
            auto make = [&](const char* name)
            {
                return graph.CreateTexture(
                    {.format = rhi::Format::RGBA16_FLOAT,
                     .width = width,
                     .height = height,
                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                     .debugName = name});
            };
            diffuseSignal = make("PathTracer.DiffuseSignal");
            specularSignal = make("PathTracer.SpecularSignal");
            blackboard.SetTexture(slots::kDiffuseSignal, diffuseSignal);
            blackboard.SetTexture(slots::kSpecularSignal, specularSignal);
        }

        ReSTIR::TraceOutputs restirOutputs;
        if (restirDi || restirGi)
        {
            restirOutputs = mReSTIR->BeginFrame(graph, width, height, !historyInvalid, restirDi, restirGi);
        }
        const rdg::BufferRef primaryHits = restirOutputs.primaryHits;
        const rdg::BufferRef giReservoirs = restirOutputs.giReservoirs;

        const rhi::PipelineHandle pipeline =
            nrc        ? nrcPipeline
            : restirGi ? (mUsedNeuralKernels ? mTracePipelineGiNeural : mTracePipelineGi)
            : march    ? mTracePipelineVolumeShadows
            : volume && mTracePipelineVolume.IsValid() ? mTracePipelineVolume
                                                       : (mUsedNeuralKernels                ? mTracePipelineNeural
                                                          : UsesShaderExecutionReordering() ? mTracePipelineSer
                                                                                            : mTracePipeline);
        RadianceCache* cache = nrc ? mRadianceCache.get() : nullptr;
        // Under real-time volumes the trace writes the frame's samples here, and the march accumulates.
        rdg::TextureRef target = accumulation;
        if (march)
        {
            target =
                graph.CreateTexture({.format = rhi::Format::RGBA32_FLOAT,
                                     .width = width,
                                     .height = height,
                                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                     .debugName = "PathTracer.FrameSamples"});
            if (mSettings.transparencyGuides && !mSettings.accumulate)
            {
                frameSamples = target;
                colorBefore = graph.CreateTexture(
                    {.format = rhi::Format::RGBA16_FLOAT, // as sceneColor: the denoiser compares the two
                     .width = width,
                     .height = height,
                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                     .debugName = "PathTracer.ColorBeforeVolumes"});
                blackboard.SetTexture(slots::kColorBeforeTransparency, colorBefore);
            }
        }
        VolumeMarcher* marcher = march ? mVolumeMarcher.get() : nullptr;
        CloudMarcher* clouds = cloudMarch ? mCloudMarcher.get() : nullptr;
        if (clouds)
        {
            clouds->AddPasses(graph, scene, mSettings.cloudMarch, constants.frameData, constants.frameSlot,
                              constants.skyTable, frame.frameIndex);
        }
        graph.AddPass(
            "PathTracer.Trace", nrc ? rdg::PassFlags::Compute : rdg::PassFlags::RayTracing,
            [&](rdg::PassBuilder& b)
            {
                b.Write(target, Access::UAVCompute);
                if (marcher)
                {
                    marcher->DeclareShadowReads(b);
                }
                if (clouds)
                {
                    clouds->DeclareReads(b);
                }
                for (rdg::BufferRef buffer : {primaryHits, giReservoirs})
                {
                    if (buffer.IsValid())
                    {
                        b.Write(buffer, Access::UAVCompute);
                    }
                }
                for (rdg::TextureRef t : {guides.diffuseAlbedo, guides.specularAlbedo, guides.normalRoughness,
                                          guides.viewDepth, guides.motionVectors, diffuseSignal, specularSignal})
                {
                    if (t.IsValid())
                    {
                        b.Write(t, Access::UAVCompute);
                    }
                }
                if (cache)
                {
                    cache->DeclareTrace(b);
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                constants.accumulation = ctx.UAV(target);
                if (guides.diffuseAlbedo.IsValid())
                {
                    constants.diffuseAlbedo = ctx.UAV(guides.diffuseAlbedo);
                    constants.specularAlbedo = ctx.UAV(guides.specularAlbedo);
                    constants.normalRoughness = ctx.UAV(guides.normalRoughness);
                    constants.viewDepth = ctx.UAV(guides.viewDepth);
                    constants.motionVectors = ctx.UAV(guides.motionVectors);
                }
                if (diffuseSignal.IsValid())
                {
                    constants.diffuseSignal = ctx.UAV(diffuseSignal);
                    constants.specularSignal = ctx.UAV(specularSignal);
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
                if (cache)
                {
                    ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
                }
                else
                {
                    ctx.cmd.DispatchRays(width, height, 1);
                }
            });

        if (march)
        {
            // What the volumes leave of the surfaces' light, for a denoiser of the surfaces.
            rdg::TextureRef surfaceTransmittance;
            if (signals)
            {
                surfaceTransmittance = graph.CreateTexture(
                    {.format = rhi::Format::R16_FLOAT,
                     .width = width,
                     .height = height,
                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                     .debugName = "PathTracer.SurfaceTransmittance"});
                blackboard.SetTexture(slots::kSurfaceTransmittance, surfaceTransmittance);
            }
            // A denoiser of the whole frame wants the guides to describe the volumes where they hide the
            // surfaces; one of the surfaces alone wants them as they are.
            const DenoiserGuides marchGuides = mSettings.transparencyGuides ? guides : DenoiserGuides{};
            VolumeMarcher::Frame marchFrame;
            marchFrame.width = width;
            marchFrame.height = height;
            marchFrame.samplesPerPixel = spp;
            marchFrame.debugView = view;
            marchFrame.reset = (frame.flags & gpu::kFlagReset) != 0u;
            marchFrame.accumulating = mSettings.accumulate;
            marchFrame.furnace = mSettings.furnace;
            // In real time the volumes' layer changes from frame to frame; a denoiser of the whole frame
            // filters it, otherwise the march does (the lit views).
            marchFrame.temporal = mSettings.volumeMarch.temporal && !mSettings.accumulate &&
                                  !mSettings.transparencyGuides &&
                                  (view == gpu::kDebugNone || view == gpu::kDebugDirect || view == gpu::kDebugIndirect);
            marchFrame.historyValid = !historyInvalid;
            mVolumeMarcher->AddPasses(graph, scene, constants, target, accumulation, marchGuides, surfaceTransmittance,
                                      marchFrame);
        }

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
            mReSTIR->AddPasses(graph, mSettings.restir, restirScene, accumulation, diffuseSignal, specularSignal,
                               view != gpu::kDebugIndirect, view != gpu::kDebugDirect, neural && scene.coatedMaterials);
        }
        if (nrc)
        {
            // The training paths' pixels get their cache lookups in the views that show indirect light.
            mRadianceCache->AddPasses(graph, constants.frameData, constants.frameSlot, accumulation,
                                      view == gpu::kDebugNone || view == gpu::kDebugIndirect);
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
            if (colorBefore.IsValid())
            {
                b.Read(frameSamples, Access::SRVCompute);
                b.Write(colorBefore, Access::UAVCompute);
            }
        },
        [=](rdg::PassContext& ctx)
        {
            gpu::ResolveConstants resolve{};
            resolve.frameData = frameData;
            resolve.frameSlot = slot;
            resolve.accumulation = ctx.SRV(accumulation);
            resolve.output = ctx.UAV(output);
            resolve.surface = colorBefore.IsValid() ? ctx.SRV(frameSamples) : air::gpu::kInvalidIndex;
            resolve.before = colorBefore.IsValid() ? ctx.UAV(colorBefore) : air::gpu::kInvalidIndex;
            ctx.cmd.SetPipeline(resolvePipeline);
            ctx.cmd.PushConstants(&resolve, sizeof(resolve));
            ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        });
}

} // namespace air::pt
