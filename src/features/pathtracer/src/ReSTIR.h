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
    bool IsDiAvailable() const { return mDiInitial.IsValid() && mDiTemporal.IsValid() && mDiSpatial.IsValid(); }
    bool IsGiAvailable() const { return mGiTemporal.IsValid() && mGiSpatial.IsValid(); }

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
    // shadeIndirect (the views that show that light).
    void AddPasses(rdg::RenderGraph& graph, const ReSTIRSettings& settings, const gpu::ReSTIRConstants& scene,
                   rdg::TextureRef accumulation, bool shadeDirect, bool shadeIndirect);

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
    rhi::PipelineHandle mDiInitial;
    rhi::PipelineHandle mDiTemporal;
    rhi::PipelineHandle mDiSpatial;
    rhi::PipelineHandle mGiTemporal;
    rhi::PipelineHandle mGiSpatial;
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
