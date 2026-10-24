#include "RadianceCache.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/Mlp.h"
#include "air/renderer/ShaderLoader.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

namespace air::pt
{

using rhi::Access;

namespace
{

constexpr u32 kRecordsPerPath = 6; // a training path's vertices that become samples
constexpr Access kRest = Access::SRVCompute;

// Hash grid levels from 16 to 512 cells across the scene's longest side.
constexpr float kCoarsest = 16.0f;
constexpr float kFinest = 512.0f;

} // namespace

RadianceCache::RadianceCache(rhi::Device& device) : mDevice(device) {}

RadianceCache::~RadianceCache()
{
    ReleaseBuffers();
    for (renderer::LazyComputePipeline* pipeline : {&mTrain, &mTrainPortable, &mResolve, &mResolvePortable})
    {
        pipeline->Destroy(mDevice);
    }
}

void RadianceCache::Initialize(const std::filesystem::path& shaderDir)
{
    mRendererShaderDir = shaderDir.parent_path() / "renderer";
    constexpr u32 kConstants = sizeof(gpu::NrcConstants);
    mTrainPortable.Load(mDevice, shaderDir, "NrcTrainPortable.cs", "NRC.TrainPortable", kConstants);
    mResolvePortable.Load(mDevice, shaderDir, "NrcResolvePortable.cs", "NRC.ResolvePortable", kConstants);
    if (mDevice.GetCapabilities().linearAlgebra)
    {
        mTrain.Load(mDevice, shaderDir, "NrcTrain.cs", "NRC.Train", kConstants);
        mResolve.Load(mDevice, shaderDir, "NrcResolve.cs", "NRC.Resolve", kConstants);
    }
}

bool RadianceCache::IsAvailable(bool linearAlgebra) const
{
    return linearAlgebra ? mTrain.IsAvailable() && mResolve.IsAvailable()
                         : mTrainPortable.IsAvailable() && mResolvePortable.IsAvailable();
}

void RadianceCache::ReleaseBuffers()
{
    for (rhi::BufferHandle* buffer : {&mRecords, &mPixelQueries})
    {
        if (buffer->IsValid())
        {
            mDevice.DestroyBuffer(*buffer);
        }
        *buffer = {};
    }
    mCapacity = 0;
}

void RadianceCache::BeginFrame(rdg::RenderGraph& graph, const renderer::GpuScene& scene,
                               const RadianceCacheSettings& settings, bool linearAlgebra, u32 width, u32 height,
                               float exposure, u32 frame, gpu::NrcData& data)
{
    mSettings = settings;
    mSettings.steps = std::max(settings.steps, 1u);

    // A network for this scene: new when the hash grid's size or the kernels' kind changed, fresh
    // parameters when the scene did.
    const u32 tableSize = 1u << std::clamp(settings.tableLog2, 10u, 22u);
    if (!mNetwork || mTableSize != tableSize || mNetworkLinearAlgebra != linearAlgebra)
    {
        mNetwork = std::make_unique<renderer::NeuralNetwork>(
            mDevice, renderer::NeuralNetworkDesc{.encodingParameters =
                                                     air::gpu::kHashLevels * tableSize * air::gpu::kHashFeatures,
                                                 .outputs = 3,
                                                 .name = "NRC.Mlp"});
        AIR_CHECK(mNetwork->Initialize(mRendererShaderDir, linearAlgebra, mSeed),
                  "neural radiance cache: the network's optimizer did not load");
        mTableSize = tableSize;
        mNetworkLinearAlgebra = linearAlgebra;
        mResetPending = false;
        mSceneVersion = scene.version;
        mNormalization = 0.0f;
    }
    if (mResetPending || scene.version != mSceneVersion)
    {
        mNetwork->Reset(++mSeed);
        mResetPending = false;
        mSceneVersion = scene.version;
        mNormalization = 0.0f;
    }
    mNetwork->BeginFrame(graph);

    // The network's outputs are in units of the exposure: when it changes, so does the output layer
    // (at this frame's first optimizer step), and this frame still predicts in the old units.
    if (mNormalization <= 0.0f)
    {
        mNormalization = exposure;
    }
    data.normalization = mNormalization;
    if (exposure != mNormalization)
    {
        mNetwork->ScaleOutputs(exposure / mNormalization);
        mNormalization = exposure;
    }

    // One training path per tile x tile pixels, at a pixel that moves every frame.
    const u32 tile =
        std::max(2u, u32(std::lround(std::sqrt(double(width) * height / std::max(settings.trainingPaths, 1u)))));
    data.tile = tile;
    data.tilesX = width / tile;
    data.tilesY = height / tile;
    mPaths = data.tilesX * data.tilesY;
    const u32 hash = (frame + 1) * 2654435761u;
    data.offsetX = (hash >> 8) % tile;
    data.offsetY = (hash >> 20) % tile;
    data.recordsPerPath = kRecordsPerPath;
    data.unbiasedPeriod = std::max(settings.unbiasedPeriod, 1u);
    data.spread = settings.spread;
    data.frame = frame;
    if (mPaths > mCapacity)
    {
        ReleaseBuffers();
        const rhi::BufferUsage usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource;
        mRecords = mDevice.CreateBuffer({.size = u64(mPaths) * kRecordsPerPath * sizeof(gpu::NrcTrainingRecord),
                                         .stride = sizeof(gpu::NrcTrainingRecord),
                                         .usage = usage,
                                         .debugName = "NRC.TrainingRecords"});
        mPixelQueries = mDevice.CreateBuffer({.size = u64(mPaths) * sizeof(gpu::NrcPixelQuery),
                                              .stride = sizeof(gpu::NrcPixelQuery),
                                              .usage = usage,
                                              .debugName = "NRC.PixelQueries"});
        mRecordsAccess = Access::None;
        mPixelQueriesAccess = Access::None;
        mCapacity = mPaths;
    }
    mRecordsRef = graph.ImportBuffer(mRecords, mRecordsAccess, kRest);
    mRecordsAccess = kRest;
    mPixelQueriesRef = graph.ImportBuffer(mPixelQueries, mPixelQueriesAccess, kRest);
    mPixelQueriesAccess = kRest;

    data.network = mNetwork->GetBuffers();
    // The grid spans the scene's box, a little larger so that its surfaces sit inside.
    const hlsl::float3 margin = (scene.boundsMax - scene.boundsMin) * 0.01f + hlsl::float3(1e-3f);
    data.grid = renderer::MakeHashGrid(scene.boundsMin - margin, scene.boundsMax + margin, tableSize, kCoarsest,
                                       kFinest, air::gpu::kMlpWeights);
    data.trainingRecords = mDevice.GetBindlessUAV(mRecords);
    data.pixelQueries = mDevice.GetBindlessUAV(mPixelQueries);
}

void RadianceCache::DeclareTrace(rdg::PassBuilder& builder) const
{
    builder.Read(mNetwork->Parameters(), Access::SRVCompute);
    if (mNetwork->Matrices().IsValid())
    {
        builder.Read(mNetwork->Matrices(), Access::SRVCompute);
    }
    builder.Write(mRecordsRef, Access::UAVCompute);
    builder.Write(mPixelQueriesRef, Access::UAVCompute);
}

void RadianceCache::AddPasses(rdg::RenderGraph& graph, u32 frameData, u32 frameSlot, rdg::TextureRef accumulation,
                              bool resolve)
{
    if (mPaths == 0)
    {
        return; // an image smaller than a tile trains nothing
    }
    const bool linearAlgebra = mNetwork->UsesLinearAlgebra();
    const rdg::BufferRef parameters = mNetwork->Parameters();
    const rdg::BufferRef matrices = mNetwork->Matrices();
    const rdg::BufferRef gradients = mNetwork->Gradients();
    const rdg::BufferRef records = mRecordsRef;
    const rdg::BufferRef pixelQueries = mPixelQueriesRef;
    gpu::NrcConstants base{};
    base.frameData = frameData;
    base.frameSlot = frameSlot;
    base.paths = mPaths;
    base.steps = mSettings.steps;

    if (resolve)
    {
        const rhi::PipelineHandle pipeline = (linearAlgebra ? mResolve : mResolvePortable).Get(mDevice);
        const u32 groups = (mPaths + 63) / 64;
        graph.AddPass(
            "NRC.Resolve", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(parameters, Access::SRVCompute);
                if (matrices.IsValid())
                {
                    b.Read(matrices, Access::SRVCompute);
                }
                b.Write(pixelQueries, Access::UAVCompute);
                b.Write(accumulation, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                gpu::NrcConstants constants = base;
                constants.accumulation = ctx.UAV(accumulation);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&constants, sizeof(constants));
                ctx.cmd.Dispatch(groups, 1, 1);
            });
    }

    // Every step trains on its share of the training paths, scattered over the image (NrcStepPath);
    // about 2.5 vertices of each become samples.
    const rhi::PipelineHandle train = (linearAlgebra ? mTrain : mTrainPortable).Get(mDevice);
    const float samples = 2.5f * float(mPaths) / float(mSettings.steps);
    u32 permutation = u32(2654435761ull % std::max(mPaths, 1u));
    while (std::gcd(permutation, mPaths) != 1)
    {
        ++permutation;
    }
    base.permutation = permutation;
    renderer::NeuralTrainingSettings training;
    training.learningRate = mSettings.learningRate;
    for (u32 step = 0; step < mSettings.steps; ++step)
    {
        const u32 slots = (mPaths - std::min(step, mPaths) + mSettings.steps - 1) / mSettings.steps * kRecordsPerPath;
        graph.AddPass(
            std::format("NRC.Train[{}]", step), rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(parameters, Access::SRVCompute);
                if (matrices.IsValid())
                {
                    b.Read(matrices, Access::SRVCompute);
                }
                b.Write(records, Access::UAVCompute);
                b.Write(gradients, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx)
            {
                gpu::NrcConstants constants = base;
                constants.step = step;
                ctx.cmd.SetPipeline(train);
                ctx.cmd.PushConstants(&constants, sizeof(constants));
                ctx.cmd.Dispatch((slots + 63) / 64, 1, 1);
            });
        mNetwork->AddOptimizerPasses(graph, training, samples);
    }
}

} // namespace air::pt
