#include "ReSTIR.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/ShaderLoader.h"

#include <algorithm>
#include <format>
#include <utility>
#include <vector>

namespace air::pt
{

using rhi::Access;

namespace
{

constexpr Access kRest = Access::SRVCompute; // where the persistent buffers rest between frames

} // namespace

ReSTIR::ReSTIR(rhi::Device& device) : mDevice(device) {}

ReSTIR::~ReSTIR()
{
    for (History* history : {&mPrimaryHits, &mDiHistory, &mGiHistory})
    {
        Release(*history);
    }
    for (const Pipelines& p : mPipelines)
    {
        for (rhi::PipelineHandle pipeline : {p.diInitial, p.diTemporal, p.diSpatial, p.giTemporal, p.giSpatial})
        {
            if (pipeline.IsValid())
            {
                mDevice.DestroyPipeline(pipeline);
            }
        }
    }
}

ReSTIR::Pipelines ReSTIR::LoadPipelines(bool coatModels)
{
    auto load = [&](const char* stem, const char* name)
    {
        rhi::ComputePipelineDesc desc;
        desc.shader =
            renderer::LoadShader(mDevice, mShaderDir, std::string(stem) + (coatModels ? "Neural.cs" : ".cs"), "Main");
        desc.pushConstantBytes = sizeof(gpu::ReSTIRConstants);
        desc.debugName = coatModels ? std::string(name) + "Neural" : std::string(name);
        return mDevice.CreateComputePipeline(desc);
    };
    Pipelines p;
    p.diInitial = load("ReSTIRDIInitial", "PathTracer.ReSTIRDI.Initial");
    p.diTemporal = load("ReSTIRDITemporal", "PathTracer.ReSTIRDI.Temporal");
    p.diSpatial = load("ReSTIRDISpatial", "PathTracer.ReSTIRDI.Spatial");
    p.giTemporal = load("ReSTIRGITemporal", "PathTracer.ReSTIRGI.Temporal");
    p.giSpatial = load("ReSTIRGISpatial", "PathTracer.ReSTIRGI.Spatial");
    return p;
}

void ReSTIR::Initialize(const std::filesystem::path& shaderDir)
{
    mShaderDir = shaderDir;
    mPipelines[0] = LoadPipelines(false);
    if (!IsDiAvailable() || !IsGiAvailable())
    {
        AIR_LOG(Warning, "PathTracer", "ReSTIR{}{} unavailable (shaders did not load)", IsDiAvailable() ? "" : " DI",
                IsGiAvailable() ? "" : " GI");
    }
}

void ReSTIR::Allocate(History& history, u32 stride, const char* name)
{
    if (history.buffers[0].IsValid())
    {
        return;
    }
    for (u32 i = 0; i < 2; ++i)
    {
        history.buffers[i] =
            mDevice.CreateBuffer({.size = u64(mWidth) * mHeight * stride,
                                  .stride = stride,
                                  .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                  .debugName = std::format("PathTracer.{}[{}]", name, i)});
    }
    history.access = {Access::None, Access::None};
    history.valid = false;
}

void ReSTIR::Release(History& history)
{
    for (rhi::BufferHandle& buffer : history.buffers)
    {
        if (buffer.IsValid())
        {
            mDevice.DestroyBuffer(buffer);
        }
        buffer = {};
    }
    history.valid = false;
}

void ReSTIR::Invalidate()
{
    mPrimaryHits.valid = false;
    mDiHistory.valid = false;
    mGiHistory.valid = false;
}

ReSTIR::TraceOutputs ReSTIR::BeginFrame(rdg::RenderGraph& graph, u32 width, u32 height, bool historyValid, bool di,
                                        bool gi)
{
    if (width != mWidth || height != mHeight)
    {
        for (History* history : {&mPrimaryHits, &mDiHistory, &mGiHistory})
        {
            Release(*history);
        }
        mWidth = width;
        mHeight = height;
    }
    // The buffers only exist once their kind of ReSTIR has been used.
    Allocate(mPrimaryHits, sizeof(gpu::PrimaryHit), "ReSTIR.PrimaryHits");
    if (di)
    {
        Allocate(mDiHistory, sizeof(gpu::Reservoir), "ReSTIRDI.History");
    }
    if (gi)
    {
        Allocate(mGiHistory, sizeof(gpu::GiReservoir), "ReSTIRGI.History");
    }
    mDi = di;
    mGi = gi;
    mUseHistory = historyValid && mPrimaryHits.valid;
    mUseDiHistory = di && mUseHistory && mDiHistory.valid;
    mUseGiHistory = gi && mUseHistory && mGiHistory.valid;

    const u32 previous = mCurrent ^ 1u;
    mHitsRef = graph.ImportBuffer(mPrimaryHits.buffers[mCurrent], mPrimaryHits.access[mCurrent], kRest);
    mPrimaryHits.access[mCurrent] = kRest;
    mPreviousHitsRef = {};
    if (mUseDiHistory || mUseGiHistory)
    {
        mPreviousHitsRef = graph.ImportBuffer(mPrimaryHits.buffers[previous], mPrimaryHits.access[previous], kRest);
    }
    mGiInitialRef = {};
    if (gi)
    {
        mGiInitialRef =
            graph.CreateBuffer({.size = u64(width) * height * sizeof(gpu::GiReservoir),
                                .stride = sizeof(gpu::GiReservoir),
                                .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                .debugName = "PathTracer.ReSTIRGI.InitialReservoirs"});
    }
    return {mHitsRef, mGiInitialRef};
}

void ReSTIR::AddPasses(rdg::RenderGraph& graph, const ReSTIRSettings& settings, const gpu::ReSTIRConstants& scene,
                       rdg::TextureRef accumulation, rdg::TextureRef diffuseSignal, rdg::TextureRef specularSignal,
                       bool shadeDirect, bool shadeIndirect, bool coatModels)
{
    AIR_CHECK(mHitsRef.IsValid(), "ReSTIR::AddPasses without BeginFrame()");
    if (coatModels && !mPipelines[1].diInitial.IsValid())
    {
        mPipelines[1] = LoadPipelines(true);
    }
    const Pipelines& p = mPipelines[coatModels && mPipelines[1].diInitial.IsValid() ? 1 : 0];
    const u32 width = mWidth;
    const u32 height = mHeight;
    const u32 groupsX = (width + 7) / 8;
    const u32 groupsY = (height + 7) / 8;
    const u32 previous = mCurrent ^ 1u;
    const rdg::BufferRef hits = mHitsRef;
    const rdg::BufferRef previousHits = mPreviousHitsRef;

    gpu::ReSTIRConstants base = scene;
    base.lightCandidates = settings.lightCandidates;
    base.bsdfCandidates = settings.bsdfCandidates;
    base.spatialSamples = std::min(settings.spatialSamples, gpu::kReSTIRMaxSpatialSamples);
    base.disocclusionSamples =
        std::min(std::max(settings.disocclusionSamples, settings.spatialSamples), gpu::kReSTIRMaxSpatialSamples);
    base.spatialRadius = settings.spatialRadius;
    base.maxConfidence = float(std::clamp(settings.maxConfidence, 1u, 254u));
    const u32 unbiased = settings.unbiased ? gpu::kReSTIRUnbiased : 0u;

    // A compute pass over the image that reads `reads`, writes `writes` (and the accumulation when
    // `shades`), with `bind` filling the pass's own buffer indices.
    auto addPass = [&](const char* name, rhi::PipelineHandle pipeline, u32 flags,
                       const std::vector<rdg::BufferRef>& reads, const std::vector<rdg::BufferRef>& writes, bool shades,
                       auto bind)
    {
        gpu::ReSTIRConstants constants = base;
        const bool signals = shades && diffuseSignal.IsValid();
        constants.flags = flags | (signals ? gpu::kReSTIRSignals : 0u);
        graph.AddPass(
            name, rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                for (rdg::BufferRef buffer : reads)
                {
                    if (buffer.IsValid())
                    {
                        b.Read(buffer, Access::SRVCompute);
                    }
                }
                for (rdg::BufferRef buffer : writes)
                {
                    b.Write(buffer, Access::UAVCompute);
                }
                if (shades)
                {
                    b.Write(accumulation, Access::UAVCompute);
                }
                if (signals)
                {
                    b.Write(diffuseSignal, Access::UAVCompute);
                    b.Write(specularSignal, Access::UAVCompute);
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                constants.primaryHits = ctx.SRV(hits);
                if (shades)
                {
                    constants.accumulation = ctx.UAV(accumulation);
                }
                if (signals)
                {
                    constants.diffuseSignal = ctx.UAV(diffuseSignal);
                    constants.specularSignal = ctx.UAV(specularSignal);
                }
                bind(ctx, constants);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&constants, sizeof(constants));
                ctx.cmd.Dispatch(groupsX, groupsY, 1);
            });
    };
    // This frame's history buffer of one kind, and last frame's when it is reused.
    auto importHistory = [&](History& history, bool useHistory)
    {
        const rdg::BufferRef current = graph.ImportBuffer(history.buffers[mCurrent], history.access[mCurrent], kRest);
        history.access[mCurrent] = kRest;
        rdg::BufferRef last;
        if (useHistory)
        {
            last = graph.ImportBuffer(history.buffers[previous], history.access[previous], kRest);
        }
        return std::pair{current, last};
    };
    auto transient = [&](u32 stride, const char* name)
    {
        return graph.CreateBuffer({.size = u64(width) * height * stride,
                                   .stride = stride,
                                   .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                   .debugName = name});
    };

    if (mDi)
    {
        const auto [history, lastHistory] = importHistory(mDiHistory, mUseDiHistory);
        const rdg::BufferRef initial = transient(sizeof(gpu::Reservoir), "PathTracer.ReSTIRDI.InitialReservoirs");
        const rdg::BufferRef temporal = transient(sizeof(gpu::Reservoir), "PathTracer.ReSTIRDI.TemporalReservoirs");
        const u32 temporalFlag = mUseDiHistory ? gpu::kReSTIRTemporal : 0u;
        addPass("PathTracer.ReSTIRDI.Initial", p.diInitial, unbiased, {hits}, {initial}, false,
                [=](rdg::PassContext& ctx, gpu::ReSTIRConstants& c) { c.output = ctx.UAV(initial); });
        addPass("PathTracer.ReSTIRDI.Temporal", p.diTemporal, unbiased | temporalFlag,
                {hits, initial, previousHits, lastHistory}, {temporal}, false,
                [=](rdg::PassContext& ctx, gpu::ReSTIRConstants& c)
                {
                    c.input = ctx.SRV(initial);
                    if (lastHistory.IsValid())
                    {
                        c.previousPrimaryHits = ctx.SRV(previousHits);
                        c.previous = ctx.SRV(lastHistory);
                    }
                    c.output = ctx.UAV(temporal);
                });
        addPass("PathTracer.ReSTIRDI.Spatial", p.diSpatial, unbiased | (shadeDirect ? gpu::kReSTIRShade : 0u),
                {hits, temporal}, {history}, shadeDirect,
                [=](rdg::PassContext& ctx, gpu::ReSTIRConstants& c)
                {
                    c.input = ctx.SRV(temporal);
                    c.output = ctx.UAV(history);
                });
    }
    if (mGi)
    {
        const auto [history, lastHistory] = importHistory(mGiHistory, mUseGiHistory);
        const rdg::BufferRef initial = mGiInitialRef;
        const rdg::BufferRef temporal = transient(sizeof(gpu::GiReservoir), "PathTracer.ReSTIRGI.TemporalReservoirs");
        const u32 temporalFlag = mUseGiHistory ? gpu::kReSTIRTemporal : 0u;
        addPass("PathTracer.ReSTIRGI.Temporal", p.giTemporal, unbiased | temporalFlag,
                {hits, initial, previousHits, lastHistory}, {temporal}, false,
                [=](rdg::PassContext& ctx, gpu::ReSTIRConstants& c)
                {
                    c.input = ctx.SRV(initial);
                    if (lastHistory.IsValid())
                    {
                        c.previousPrimaryHits = ctx.SRV(previousHits);
                        c.previous = ctx.SRV(lastHistory);
                    }
                    c.output = ctx.UAV(temporal);
                });
        addPass("PathTracer.ReSTIRGI.Spatial", p.giSpatial, unbiased | (shadeIndirect ? gpu::kReSTIRShade : 0u),
                {hits, temporal}, {history}, shadeIndirect,
                [=](rdg::PassContext& ctx, gpu::ReSTIRConstants& c)
                {
                    c.input = ctx.SRV(temporal);
                    c.output = ctx.UAV(history);
                });
    }

    // This frame's buffers become next frame's history.
    mPrimaryHits.valid = true;
    mDiHistory.valid = mDi;
    mGiHistory.valid = mGi;
    mCurrent ^= 1u;
    mHitsRef = {};
    mPreviousHitsRef = {};
    mGiInitialRef = {};
}

} // namespace air::pt
