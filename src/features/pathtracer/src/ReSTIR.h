#pragma once

// ReSTIR DI passes of the path tracer (shaders/ReSTIR.hlsli): after PathTracer.Trace, three compute
// passes resample the direct light at each pixel's first surface across pixels and frames, and add
// it to the accumulation. Owns the buffers that live across frames: this and last frame's primary
// hits and final reservoirs.

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

    // Loads ReSTIRInitial/Temporal/Spatial.cs from shaderDir. False: ReSTIR is unavailable.
    bool Initialize(const std::filesystem::path& shaderDir);
    bool IsAvailable() const { return mInitial.IsValid() && mTemporal.IsValid() && mSpatial.IsValid(); }

    // Starts a frame of width x height pixels and returns the primary-hit buffer PathTracer.Trace
    // writes (RWStructuredBuffer<gpu::PrimaryHit>). Last frame's hits and reservoirs are reused
    // unless `historyValid` is false (scene or settings changed) or the last frame had no ReSTIR.
    rdg::BufferRef BeginFrame(rdg::RenderGraph& graph, u32 width, u32 height, bool historyValid);

    // Adds the three passes. `scene` carries the trace pass's frame and scene indices; the spatial
    // pass adds the direct light into `accumulation` when addDirect (lit and direct-light views).
    void AddPasses(rdg::RenderGraph& graph, const ReSTIRSettings& settings, const gpu::ReSTIRConstants& scene,
                   rdg::TextureRef accumulation, bool addDirect);

    // A frame without ReSTIR: the next one starts without history.
    void Invalidate() { mHistoryValid = false; }

private:
    rhi::Device& mDevice;
    rhi::PipelineHandle mInitial;
    rhi::PipelineHandle mTemporal;
    rhi::PipelineHandle mSpatial;
    // Indexed by frame parity: this frame writes [mCurrent], reads last frame's [mCurrent ^ 1].
    std::array<rhi::BufferHandle, 2> mPrimaryHits;
    std::array<rhi::BufferHandle, 2> mHistory; // final reservoirs
    std::array<rhi::Access, 2> mHitsAccess{rhi::Access::None, rhi::Access::None};
    std::array<rhi::Access, 2> mHistoryAccess{rhi::Access::None, rhi::Access::None};
    u32 mWidth = 0;
    u32 mHeight = 0;
    u32 mCurrent = 0;
    bool mHistoryValid = false;

    // This frame's graph resources, from BeginFrame.
    rdg::BufferRef mHitsRef;
    rdg::BufferRef mPreviousHitsRef;
    rdg::BufferRef mPreviousHistoryRef;
    bool mUseHistory = false;
};

} // namespace air::pt
