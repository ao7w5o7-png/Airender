#include "ReSTIR.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/ShaderLoader.h"

#include <algorithm>
#include <format>

namespace air::pt
{

using rhi::Access;

ReSTIR::ReSTIR(rhi::Device& device) : mDevice(device) {}

ReSTIR::~ReSTIR()
{
    for (rhi::BufferHandle buffer : {mPrimaryHits[0], mPrimaryHits[1], mHistory[0], mHistory[1]})
    {
        if (buffer.IsValid())
        {
            mDevice.DestroyBuffer(buffer);
        }
    }
    for (rhi::PipelineHandle pipeline : {mInitial, mTemporal, mSpatial})
    {
        if (pipeline.IsValid())
        {
            mDevice.DestroyPipeline(pipeline);
        }
    }
}

bool ReSTIR::Initialize(const std::filesystem::path& shaderDir)
{
    auto load = [&](const char* stem, const char* name)
    {
        rhi::ComputePipelineDesc desc;
        desc.shader = renderer::LoadShader(mDevice, shaderDir, stem, "Main");
        desc.pushConstantBytes = sizeof(gpu::ReSTIRConstants);
        desc.debugName = name;
        return mDevice.CreateComputePipeline(desc);
    };
    mInitial = load("ReSTIRInitial.cs", "PathTracer.ReSTIR.Initial");
    mTemporal = load("ReSTIRTemporal.cs", "PathTracer.ReSTIR.Temporal");
    mSpatial = load("ReSTIRSpatial.cs", "PathTracer.ReSTIR.Spatial");
    if (!IsAvailable())
    {
        AIR_LOG(Warning, "PathTracer", "ReSTIR DI unavailable (its shaders did not load)");
    }
    return IsAvailable();
}

rdg::BufferRef ReSTIR::BeginFrame(rdg::RenderGraph& graph, u32 width, u32 height, bool historyValid)
{
    if (width != mWidth || height != mHeight || !mPrimaryHits[0].IsValid())
    {
        // First use (the buffers only exist while ReSTIR is used) or a new size.
        for (u32 i = 0; i < 2; ++i)
        {
            for (rhi::BufferHandle* buffer : {&mPrimaryHits[i], &mHistory[i]})
            {
                if (buffer->IsValid())
                {
                    mDevice.DestroyBuffer(*buffer);
                }
            }
            const u64 pixels = u64(width) * height;
            mPrimaryHits[i] =
                mDevice.CreateBuffer({.size = pixels * sizeof(gpu::PrimaryHit),
                                      .stride = sizeof(gpu::PrimaryHit),
                                      .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                      .debugName = std::format("PathTracer.ReSTIR.PrimaryHits[{}]", i)});
            mHistory[i] =
                mDevice.CreateBuffer({.size = pixels * sizeof(gpu::Reservoir),
                                      .stride = sizeof(gpu::Reservoir),
                                      .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                      .debugName = std::format("PathTracer.ReSTIR.History[{}]", i)});
        }
        mHitsAccess = {Access::None, Access::None};
        mHistoryAccess = {Access::None, Access::None};
        mWidth = width;
        mHeight = height;
        mHistoryValid = false;
    }
    mUseHistory = historyValid && mHistoryValid;
    // Between frames the buffers rest as SRVCompute.
    const Access rest = Access::SRVCompute;
    mHitsRef = graph.ImportBuffer(mPrimaryHits[mCurrent], mHitsAccess[mCurrent], rest);
    mHitsAccess[mCurrent] = rest;
    mPreviousHitsRef = {};
    mPreviousHistoryRef = {};
    if (mUseHistory)
    {
        const u32 previous = mCurrent ^ 1u;
        mPreviousHitsRef = graph.ImportBuffer(mPrimaryHits[previous], mHitsAccess[previous], rest);
        mPreviousHistoryRef = graph.ImportBuffer(mHistory[previous], mHistoryAccess[previous], rest);
    }
    return mHitsRef;
}

void ReSTIR::AddPasses(rdg::RenderGraph& graph, const ReSTIRSettings& settings, const gpu::ReSTIRConstants& scene,
                       rdg::TextureRef accumulation, bool addDirect)
{
    AIR_CHECK(mHitsRef.IsValid(), "ReSTIR::AddPasses without BeginFrame()");
    const u32 width = mWidth;
    const u32 height = mHeight;
    const Access rest = Access::SRVCompute;
    const rdg::BufferRef hits = mHitsRef;
    const rdg::BufferRef previousHits = mPreviousHitsRef;
    const rdg::BufferRef previousHistory = mPreviousHistoryRef;
    const rdg::BufferRef history = graph.ImportBuffer(mHistory[mCurrent], mHistoryAccess[mCurrent], rest);
    mHistoryAccess[mCurrent] = rest;
    auto reservoirs = [&](const char* name)
    {
        return graph.CreateBuffer({.size = u64(width) * height * sizeof(gpu::Reservoir),
                                   .stride = sizeof(gpu::Reservoir),
                                   .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                   .debugName = name});
    };
    const rdg::BufferRef initial = reservoirs("PathTracer.ReSTIR.InitialReservoirs");
    const rdg::BufferRef temporal = reservoirs("PathTracer.ReSTIR.TemporalReservoirs");

    gpu::ReSTIRConstants constants = scene;
    constants.flags = (settings.unbiased ? gpu::kReSTIRUnbiased : 0u) | (mUseHistory ? gpu::kReSTIRTemporal : 0u) |
                      (addDirect ? gpu::kReSTIRAddDirect : 0u);
    constants.lightCandidates = settings.lightCandidates;
    constants.bsdfCandidates = settings.bsdfCandidates;
    constants.spatialSamples = std::min(settings.spatialSamples, gpu::kReSTIRMaxSpatialSamples);
    constants.disocclusionSamples =
        std::min(std::max(settings.disocclusionSamples, settings.spatialSamples), gpu::kReSTIRMaxSpatialSamples);
    constants.spatialRadius = settings.spatialRadius;
    constants.maxConfidence = float(std::clamp(settings.maxConfidence, 1u, 254u));
    const u32 groupsX = (width + 7) / 8;
    const u32 groupsY = (height + 7) / 8;

    const rhi::PipelineHandle initialPipeline = mInitial;
    graph.AddPass(
        "PathTracer.ReSTIR.Initial", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(hits, Access::SRVCompute);
            b.Write(initial, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.primaryHits = ctx.SRV(hits);
            constants.output = ctx.UAV(initial);
            ctx.cmd.SetPipeline(initialPipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch(groupsX, groupsY, 1);
        });

    const rhi::PipelineHandle temporalPipeline = mTemporal;
    graph.AddPass(
        "PathTracer.ReSTIR.Temporal", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(hits, Access::SRVCompute);
            b.Read(initial, Access::SRVCompute);
            if (previousHits.IsValid())
            {
                b.Read(previousHits, Access::SRVCompute);
                b.Read(previousHistory, Access::SRVCompute);
            }
            b.Write(temporal, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.primaryHits = ctx.SRV(hits);
            constants.input = ctx.SRV(initial);
            if (previousHits.IsValid())
            {
                constants.previousPrimaryHits = ctx.SRV(previousHits);
                constants.previous = ctx.SRV(previousHistory);
            }
            constants.output = ctx.UAV(temporal);
            ctx.cmd.SetPipeline(temporalPipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch(groupsX, groupsY, 1);
        });

    const rhi::PipelineHandle spatialPipeline = mSpatial;
    graph.AddPass(
        "PathTracer.ReSTIR.Spatial", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(hits, Access::SRVCompute);
            b.Read(temporal, Access::SRVCompute);
            b.Write(history, Access::UAVCompute);
            b.Write(accumulation, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.primaryHits = ctx.SRV(hits);
            constants.input = ctx.SRV(temporal);
            constants.output = ctx.UAV(history);
            constants.accumulation = ctx.UAV(accumulation);
            ctx.cmd.SetPipeline(spatialPipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch(groupsX, groupsY, 1);
        });

    // This frame's buffers become next frame's history.
    mCurrent ^= 1u;
    mHistoryValid = true;
    mHitsRef = {};
}

} // namespace air::pt
