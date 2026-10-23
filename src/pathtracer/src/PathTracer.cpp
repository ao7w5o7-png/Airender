#include "air/pathtracer/PathTracer.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include "pathtracer/PathTracerShared.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace air::pt {

using rhi::Access;

namespace {

rhi::ShaderBlob LoadShader(const std::filesystem::path& path, std::string entry, bool required) {
    rhi::ShaderBlob blob;
    blob.entryPoint = std::move(entry);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (required) {
            AIR_LOG(Error, "PathTracer", "missing shader {}", path.string());
        }
        return blob;
    }
    blob.bytecode.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return blob;
}

void Normalize(float v[3]) {
    const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 0) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

void Cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

} // namespace

PathTracer::PathTracer(rhi::Device& device, const PathTracerSettings& settings)
    : mDevice(device), mSettings(settings) {}

PathTracer::~PathTracer() {
    if (mAccumulation.IsValid()) {
        mDevice.DestroyTexture(mAccumulation);
    }
    if (mTracePipeline.IsValid()) {
        mDevice.DestroyPipeline(mTracePipeline);
    }
    if (mResolvePipeline.IsValid()) {
        mDevice.DestroyPipeline(mResolvePipeline);
    }
}

bool PathTracer::Initialize(const std::filesystem::path& shaderDir) {
    if (!mDevice.GetCapabilities().rayTracing) {
        AIR_LOG(Error, "PathTracer", "device '{}' does not support hardware ray tracing",
                mDevice.GetCapabilities().adapterName);
        return false;
    }
    const bool required = mDevice.GetBackend() != rhi::Backend::Null;
    const char* ext = mDevice.GetBackend() == rhi::Backend::Vulkan ? ".spv" : ".dxil";

    rhi::RayTracingPipelineDesc trace;
    trace.library = LoadShader(shaderDir / (std::string("PathTracer.lib") + ext), "", required);
    trace.rayGen = "RayGen";
    trace.missShaders = {"Miss"};
    trace.hitGroups = {{"HitGroup", "ClosestHit", ""}};
    trace.maxPayloadBytes = 32;
    trace.maxRecursionDepth = 1;
    trace.pushConstantBytes = sizeof(TraceConstants);
    trace.debugName = "PathTracer.Trace";
    mTracePipeline = mDevice.CreateRayTracingPipeline(trace);

    rhi::ComputePipelineDesc resolve;
    resolve.shader = LoadShader(shaderDir / (std::string("Resolve.cs") + ext), "Main", required);
    resolve.pushConstantBytes = sizeof(ResolveConstants);
    resolve.debugName = "PathTracer.Resolve";
    mResolvePipeline = mDevice.CreateComputePipeline(resolve);

    rhi::TextureDesc accumulation;
    accumulation.format = rhi::Format::RGBA32_FLOAT;
    accumulation.width = mSettings.width;
    accumulation.height = mSettings.height;
    accumulation.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
    accumulation.debugName = "PathTracer.Accumulation";
    mAccumulation = mDevice.CreateTexture(accumulation);
    mAccumulationAccess = Access::None;

    return mTracePipeline.IsValid() && mResolvePipeline.IsValid() && mAccumulation.IsValid();
}

void PathTracer::SetCamera(const Camera& camera) {
    mCamera = camera;
    ResetAccumulation();
}

void PathTracer::ResetAccumulation() {
    mResetPending = true;
    mAccumulatedSamples = 0;
}

void PathTracer::AddPasses(rdg::RenderGraph& graph, const PathTracerScene& scene, rdg::TextureRef output) {
    AIR_CHECK(mAccumulation.IsValid(), "PathTracer::AddPasses before Initialize()");

    // The accumulation buffer persists across frames: import it and remember the state it ends in.
    const Access endAccess = Access::SRVCompute;
    rdg::TextureRef accumulation = graph.ImportTexture(mAccumulation, mAccumulationAccess, endAccess);
    mAccumulationAccess = endAccess;

    TraceConstants trace{};
    trace.tlas = mDevice.GetBindlessSRV(scene.tlas);
    trace.instanceBuffer = mDevice.GetBindlessSRV(scene.instances);
    trace.materialBuffer = mDevice.GetBindlessSRV(scene.materials);
    trace.frameIndex = mFrameIndex++;
    trace.samplesPerPixel = mSettings.samplesPerPixelPerFrame;
    trace.maxBounces = mSettings.maxBounces;
    trace.resetAccumulation = mResetPending ? 1u : 0u;

    float forward[3] = {mCamera.forward[0], mCamera.forward[1], mCamera.forward[2]};
    Normalize(forward);
    float right[3], up[3];
    Cross(forward, mCamera.up, right);
    Normalize(right);
    Cross(right, forward, up);
    trace.cameraPositionFov = {mCamera.position[0], mCamera.position[1], mCamera.position[2], mCamera.verticalFovRadians};
    trace.cameraForward = {forward[0], forward[1], forward[2], 0};
    trace.cameraRight = {right[0], right[1], right[2], 0};
    trace.cameraUp = {up[0], up[1], up[2], 0};

    mResetPending = false;
    mAccumulatedSamples += mSettings.samplesPerPixelPerFrame;

    const u32 width = mSettings.width;
    const u32 height = mSettings.height;
    const rhi::PipelineHandle tracePipeline = mTracePipeline;
    const rhi::PipelineHandle resolvePipeline = mResolvePipeline;

    graph.AddPass(
        "PathTracer.Trace", rdg::PassFlags::RayTracing,
        [&](rdg::PassBuilder& b) { b.Write(accumulation, Access::UAVCompute); },
        [=](rdg::PassContext& ctx) mutable {
            trace.accumulationUav = ctx.UAV(accumulation);
            ctx.cmd.SetPipeline(tracePipeline);
            ctx.cmd.PushConstants(&trace, sizeof(trace));
            ctx.cmd.DispatchRays(width, height, 1);
        });

    const float exposure = mSettings.exposure;
    graph.AddPass(
        "PathTracer.Resolve", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b) {
            b.Read(accumulation, Access::SRVCompute);
            b.Write(output, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) {
            ResolveConstants constants{};
            constants.accumulationSrv = ctx.SRV(accumulation);
            constants.outputUav = ctx.UAV(output);
            constants.exposure = exposure;
            constants.width = width;
            constants.height = height;
            ctx.cmd.SetPipeline(resolvePipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        });
}

} // namespace air::pt
