#pragma once

// The neural radiance cache of the path tracer (shaders/NrcShared.h, docs/architecture.md §8.11):
// the network (renderer::NeuralNetwork), the training records and the passes around
// PathTracer.Trace:
//
//   PathTracer.Trace       paths end in the cache (predicted inline), training paths write records
//   NRC.Resolve            the training paths' deferred render queries into the accumulation
//   NRC.Train[i]           one optimizer step's records: gradients
//   NRC.Mlp.*              the network's optimizer step (renderer::NeuralNetwork)

#include "air/features/pathtracer/PathTracer.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/NeuralNetwork.h"
#include "air/renderer/ShaderLoader.h"
#include "air/rhi/RHIDevice.h"

#include "PathTracerShared.h"

#include <filesystem>
#include <memory>

namespace air::pt
{

class RadianceCache
{
public:
    explicit RadianceCache(rhi::Device& device);
    ~RadianceCache();

    RadianceCache(const RadianceCache&) = delete;
    RadianceCache& operator=(const RadianceCache&) = delete;

    // Loads NrcTrain*.cs and NrcResolve*.cs from shaderDir (their pipelines are created the first time
    // the cache runs), the optimizer from <shaderDir>/../renderer.
    void Initialize(const std::filesystem::path& shaderDir);
    // The kernels of the linear algebra or the float32 path exist (the trace kernel is the caller's).
    bool IsAvailable(bool linearAlgebra) const;

    // Starts a frame of width x height pixels: (re)creates the network for a new scene or size, imports
    // its buffers and fills `data` (FrameData::nrc). `exposure` normalizes the network's outputs.
    void BeginFrame(rdg::RenderGraph& graph, const renderer::GpuScene& scene, const RadianceCacheSettings& settings,
                    bool linearAlgebra, u32 width, u32 height, float exposure, u32 frame, gpu::NrcData& data);
    // What PathTracer.Trace reads (parameters, matrices) and writes (records, pixel queries).
    void DeclareTrace(rdg::PassBuilder& builder) const;
    // After the trace: the training paths' render queries into `accumulation` (when `resolve`), then
    // the frame's optimizer steps.
    void AddPasses(rdg::RenderGraph& graph, u32 frameData, u32 frameSlot, rdg::TextureRef accumulation, bool resolve);
    // The next frame starts from a fresh network.
    void Invalidate() { mResetPending = true; }

    bool UsesLinearAlgebra() const { return mNetwork && mNetwork->UsesLinearAlgebra(); }
    u32 GetTrainingPaths() const { return mPaths; }
    u32 GetStep() const { return mNetwork ? mNetwork->GetStep() : 0; }

private:
    void ReleaseBuffers();

    rhi::Device& mDevice;
    std::filesystem::path mRendererShaderDir;
    renderer::LazyComputePipeline mTrain;           // NrcTrain.cs (linear algebra)
    renderer::LazyComputePipeline mTrainPortable;   // NrcTrainPortable.cs
    renderer::LazyComputePipeline mResolve;         // NrcResolve.cs (linear algebra)
    renderer::LazyComputePipeline mResolvePortable; // NrcResolvePortable.cs

    std::unique_ptr<renderer::NeuralNetwork> mNetwork;
    bool mNetworkLinearAlgebra = false;
    u32 mTableSize = 0;
    u64 mSceneVersion = ~0ull;
    bool mResetPending = true;
    u32 mSeed = 1;
    float mNormalization = 0.0f; // the exposure the network's outputs are in

    rhi::BufferHandle mRecords;      // RWStructuredBuffer<gpu::NrcTrainingRecord>
    rhi::BufferHandle mPixelQueries; // RWStructuredBuffer<gpu::NrcPixelQuery>
    rhi::Access mRecordsAccess = rhi::Access::None;
    rhi::Access mPixelQueriesAccess = rhi::Access::None;
    u32 mPaths = 0;
    u32 mCapacity = 0; // training paths the buffers hold

    // This frame, from BeginFrame.
    RadianceCacheSettings mSettings;
    rdg::BufferRef mRecordsRef;
    rdg::BufferRef mPixelQueriesRef;
};

} // namespace air::pt
