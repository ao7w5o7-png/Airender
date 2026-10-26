#pragma once

// ReSTIR passes of the path tracer: after PathTracer.Trace, compute passes resample the direct light
// (ReSTIR DI, shaders/ReSTIRDI.hlsli) and the indirect light (ReSTIR GI, shaders/ReSTIRGI.hlsli) at
// each pixel's first surface across pixels and frames, and add them to the accumulation. Owns the
// buffers that live across frames: this and last frame's primary hits and final reservoirs.

#include "air/features/pathtracer/PathTracer.h"
#include "air/rdg/RenderGraph.h"
#include "air/rhi/RHIDevice.h"

#include "ReSTIRShared.h"

#include <array>
#include <filesystem>

namespace air::pt
{

class ReSTIR
{
public:
    explicit ReSTIR(rhi::Device& device);
    ~ReSTIR();

    ReSTIR(const ReSTIR&) = delete;
    ReSTIR& operator=(const ReSTIR&) = delete;

    // Loads the ReSTIRDI*.cs and ReSTIRGI*.cs shaders from shaderDir.
    void Initialize(const std::filesystem::path& shaderDir);
    bool IsDiAvailable() const
    {
        const Pipelines& p = mPipelines[0];
        return p.diInitial.IsValid() && p.diTemporal.IsValid() && p.diSpatial.IsValid();
    }
    bool IsGiAvailable() const { return mPipelines[0].giTemporal.IsValid() && mPipelines[0].giSpatial.IsValid(); }

    // What PathTracer.Trace writes for the passes this frame.
    struct TraceOutputs
    {
        rdg::BufferRef primaryHits;  // RWStructuredBuffer<gpu::PrimaryHit>
        rdg::BufferRef giReservoirs; // RWStructuredBuffer<gpu::GiReservoir>, when gi
    };

    // Starts a frame of width x height pixels with ReSTIR DI and/or GI. Last frame's hits and
    // reservoirs are reused unless `historyValid` is false (scene or settings changed) or the last
    // frame did not run the same kind.
    TraceOutputs BeginFrame(rdg::RenderGraph& graph, u32 width, u32 height, bool historyValid, bool di, bool gi);

    // Adds the passes of the kinds BeginFrame started. `scene` carries the trace pass's frame and scene
    // indices. The spatial passes add their estimates into `accumulation` when shadeDirect /
    // shadeIndirect (the views that show that light), and split by lobe into the denoiser signals when
    // they are valid (slots::kDiffuseSignal, kSpecularSignal).
    void AddPasses(rdg::RenderGraph& graph, const ReSTIRSettings& settings, const gpu::ReSTIRConstants& scene,
                   rdg::TextureRef accumulation, rdg::TextureRef diffuseSignal, rdg::TextureRef specularSignal,
                   bool shadeDirect, bool shadeIndirect, bool coatModels);

    // A frame without ReSTIR: the next one starts without history.
    void Invalidate();

private:
    // Two buffers that swap every frame: this frame writes [current], last frame's is [current ^ 1].
    struct History
    {
        std::array<rhi::BufferHandle, 2> buffers;
        std::array<rhi::Access, 2> access{rhi::Access::None, rhi::Access::None};
        bool valid = false; // [current ^ 1] holds last frame's data
    };

    void Allocate(History& history, u32 stride, const char* name);
    void Release(History& history);

    rhi::Device& mDevice;
    // The passes' pipelines: [0] the plain ones, [1] with the clear coats' random walk and network
    // (ReSTIR*Neural.hlsl, AIR_PT_NEURAL: what the trace's neural kernels shade coated materials with),
    // created the first time a scene with coated materials comes.
    struct Pipelines
    {
        rhi::PipelineHandle diInitial;
        rhi::PipelineHandle diTemporal;
        rhi::PipelineHandle diSpatial;
        rhi::PipelineHandle giTemporal;
        rhi::PipelineHandle giSpatial;
    };
    Pipelines LoadPipelines(bool coatModels);
    std::filesystem::path mShaderDir;
    Pipelines mPipelines[2];
    History mPrimaryHits;
    History mDiHistory; // final reservoirs
    History mGiHistory;
    u32 mWidth = 0;
    u32 mHeight = 0;
    u32 mCurrent = 0;

    // This frame, from BeginFrame.
    bool mDi = false;
    bool mGi = false;
    bool mUseHistory = false;
    bool mUseDiHistory = false;
    bool mUseGiHistory = false;
    rdg::BufferRef mHitsRef;
    rdg::BufferRef mPreviousHitsRef;
    rdg::BufferRef mGiInitialRef;
};

} // namespace air::pt
