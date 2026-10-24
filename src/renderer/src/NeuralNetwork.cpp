#include "air/renderer/NeuralNetwork.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/Mlp.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace air::renderer
{

using rhi::Access;

namespace
{

u64 Align128(u64 value)
{
    return (value + 127) & ~u64(127);
}

// Where the persistent buffers rest between frames.
constexpr Access kReadRest = Access::SRVCompute;
constexpr Access kWriteRest = Access::UAVCompute;

} // namespace

NeuralNetwork::NeuralNetwork(rhi::Device& device, NeuralNetworkDesc desc)
    : mDevice(device), mDesc(std::move(desc)), mParameterCount(gpu::kMlpWeights + mDesc.encodingParameters)
{
    AIR_CHECK(mParameterCount % 2 == 0, "{}: {} parameters, the optimizer takes them in pairs", mDesc.name,
              mParameterCount);
}

NeuralNetwork::~NeuralNetwork()
{
    ReleaseBuffers();
    if (mOptimizer.IsValid())
    {
        mDevice.DestroyPipeline(mOptimizer);
    }
}

std::vector<float> NeuralNetwork::ReadParameters()
{
    const std::vector<u8> bytes = ReadbackBuffer(mDevice, mParameters, mParametersAccess);
    std::vector<float> parameters(mParameterCount, 0.0f);
    std::memcpy(parameters.data(), bytes.data(), std::min<size_t>(bytes.size(), parameters.size() * 4));
    return parameters;
}

std::vector<float> NeuralNetwork::ReadWeightGradients()
{
    std::vector<float> gradients(gpu::kMlpWeights, 0.0f);
    if (mLinearAlgebra)
    {
        // The device's layout to row major first.
        rdg::RenderGraph graph(mDevice);
        const rdg::BufferRef source = graph.ImportBuffer(mGradients, mGradientsAccess, mGradientsAccess);
        const rdg::BufferRef rowMajor = graph.ImportBuffer(mGradientsRowMajor, mGradientsRowMajorAccess, kReadRest);
        mGradientsRowMajorAccess = kReadRest;
        const gpu::MlpBuffers buffers = mBuffers;
        graph.AddPass(
            mDesc.name + ".ReadGradients", rdg::PassFlags::Compute | rdg::PassFlags::NeverCull,
            [&](rdg::PassBuilder& b)
            {
                b.Read(source, Access::SRVCompute);
                b.Write(rowMajor, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx)
            {
                rhi::MatrixConversion conversions[gpu::kMlpMatrices];
                for (u32 m = 0; m < gpu::kMlpMatrices; ++m)
                {
                    conversions[m] = {.rows = gpu::kMlpMatrixRows[m],
                                      .columns = gpu::kMlpWidth,
                                      .src = ctx.Get(source),
                                      .srcOffset = gpu::MlpGradientByteOffset(buffers, m),
                                      .srcLayout = rhi::MatrixLayout::OuterProductOptimal,
                                      .srcType = rhi::MatrixType::Float32,
                                      .dst = ctx.Get(rowMajor),
                                      .dstOffset = u64(gpu::MlpMatrixOffset(m)) * 4,
                                      .dstLayout = rhi::MatrixLayout::RowMajor,
                                      .dstType = rhi::MatrixType::Float32,
                                      .dstStride = gpu::kMlpWidth * 4};
                }
                ctx.cmd.ConvertMatrices(conversions);
            });
        mDevice.WaitForFence(rhi::QueueType::Graphics, graph.Execute());
        const std::vector<u8> bytes = ReadbackBuffer(mDevice, mGradientsRowMajor, mGradientsRowMajorAccess);
        std::memcpy(gradients.data(), bytes.data(), std::min<size_t>(bytes.size(), gradients.size() * 4));
        return gradients;
    }
    // Row-major 64-bit fixed point.
    const std::vector<u8> bytes = ReadbackBuffer(mDevice, mGradients, mGradientsAccess);
    for (u32 i = 0; i < gpu::kMlpWeights && (i + 1) * 8 <= bytes.size(); ++i)
    {
        i64 fixed = 0;
        std::memcpy(&fixed, bytes.data() + size_t(i) * 8, 8);
        gradients[i] = float(double(fixed) / double(gpu::kGradientFixedPoint));
    }
    return gradients;
}

bool NeuralNetwork::Initialize(const std::filesystem::path& rendererShaderDir, bool linearAlgebra, u32 seed)
{
    mLinearAlgebra = linearAlgebra && mDevice.GetCapabilities().linearAlgebra;
    rhi::ComputePipelineDesc optimize;
    optimize.shader = LoadShader(mDevice, rendererShaderDir, "MlpOptimize.cs", "Main");
    optimize.pushConstantBytes = sizeof(gpu::MlpOptimizeConstants);
    optimize.debugName = mDesc.name + ".Optimize";
    mOptimizer = mDevice.CreateComputePipeline(optimize);
    if (!mOptimizer.IsValid())
    {
        AIR_LOG(Error, "Renderer", "{}: MlpOptimize.cs did not load", mDesc.name);
        return false;
    }
    Reset(seed);
    return true;
}

void NeuralNetwork::Reset(u32 seed)
{
    Upload(InitializeMlpParameters(mDesc.encodingParameters, mDesc.outputs, seed));
}

void NeuralNetwork::Upload(const std::vector<float>& parameters)
{
    AIR_CHECK(parameters.size() == mParameterCount, "{}: {} parameters uploaded, the network has {}", mDesc.name,
              parameters.size(), mParameterCount);
    ReleaseBuffers();
    CreateBuffers(parameters);
    mStep = 0;
    mOutputScale = 1.0f;
}

void NeuralNetwork::CreateBuffers(const std::vector<float>& parameters)
{
    const rhi::BufferUsage usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource;
    auto create = [&](u64 size, const char* what, const void* data)
    {
        std::vector<u8> zeros;
        if (!data)
        {
            zeros.assign(size, 0);
            data = zeros.data();
        }
        return CreateBufferWithData(mDevice, {.size = size, .usage = usage, .debugName = mDesc.name + "." + what}, data,
                                    size);
    };
    mParameters = create(u64(mParameterCount) * 4, "Parameters", parameters.data());
    mMoments = create(u64(mParameterCount) * 8, "Moments", nullptr);

    // The MLP's gradients first (each matrix 128-byte aligned for the linear algebra paths), then the
    // encoding's as int64 fixed point.
    mBuffers = {};
    u64 gradientBytes = 0;
    u64 matrixBytes = 0;
    for (u32 m = 0; m < gpu::kMlpMatrices; ++m)
    {
        const u32 rows = gpu::kMlpMatrixRows[m];
        u64 gradientSize = u64(rows) * gpu::kMlpWidth * 8;
        if (mLinearAlgebra)
        {
            gradientSize = mDevice.GetMatrixSize(rows, gpu::kMlpWidth, rhi::MatrixType::Float32,
                                                 rhi::MatrixLayout::OuterProductOptimal);
            const u32 offset = u32(matrixBytes);
            (m == 0 ? mBuffers.matrixOffset0 : m == 1 ? mBuffers.matrixOffset1 : mBuffers.matrixOffset2) = offset;
            matrixBytes = Align128(matrixBytes + mDevice.GetMatrixSize(rows, gpu::kMlpWidth, rhi::MatrixType::Float16,
                                                                       rhi::MatrixLayout::MulOptimal));
        }
        (m == 0   ? mBuffers.gradientOffset0
         : m == 1 ? mBuffers.gradientOffset1
                  : mBuffers.gradientOffset2) = u32(gradientBytes);
        gradientBytes = mLinearAlgebra ? Align128(gradientBytes + gradientSize) : gradientBytes + gradientSize;
    }
    mMatrixGradientBytes = mLinearAlgebra ? u32(gradientBytes) : 0;
    mBuffers.encodingGradients = u32(gradientBytes);
    gradientBytes += u64(mDesc.encodingParameters) * 8;
    mGradients = create(gradientBytes, "Gradients", nullptr);
    mParametersAccess = kReadRest;
    mMomentsAccess = kReadRest;
    mGradientsAccess = kReadRest;

    if (mLinearAlgebra)
    {
        std::vector<u16> half(gpu::kMlpWeights);
        std::transform(parameters.begin(), parameters.begin() + gpu::kMlpWeights, half.begin(), FloatToHalf);
        mHalfWeights = create(half.size() * 2, "HalfWeights", half.data());
        mMatrices = create(matrixBytes, "Matrices", nullptr);
        mGradientsRowMajor = create(u64(gpu::kMlpWeights) * 4, "GradientsRowMajor", nullptr);
        mHalfWeightsAccess = kReadRest;
        mMatricesAccess = kReadRest;
        mGradientsRowMajorAccess = kReadRest;
        mConvertPending = true;
    }
    mBuffers.parameters = mDevice.GetBindlessSRV(mParameters);
    mBuffers.gradients = mDevice.GetBindlessUAV(mGradients);
    mBuffers.linearAlgebra = mLinearAlgebra ? 1u : 0u;
    if (mLinearAlgebra)
    {
        mBuffers.matrices = mDevice.GetBindlessSRV(mMatrices);
    }
    else
    {
        // Row-major int64 per weight.
        mBuffers.gradientOffset0 = gpu::MlpMatrixOffset(0) * 8;
        mBuffers.gradientOffset1 = gpu::MlpMatrixOffset(1) * 8;
        mBuffers.gradientOffset2 = gpu::MlpMatrixOffset(2) * 8;
    }
}

void NeuralNetwork::ReleaseBuffers()
{
    for (rhi::BufferHandle* buffer :
         {&mParameters, &mMoments, &mGradients, &mGradientsRowMajor, &mHalfWeights, &mMatrices})
    {
        if (buffer->IsValid())
        {
            mDevice.DestroyBuffer(*buffer);
        }
        *buffer = {};
    }
}

void NeuralNetwork::BeginFrame(rdg::RenderGraph& graph)
{
    AIR_CHECK(mParameters.IsValid(), "{}: BeginFrame before Initialize()", mDesc.name);
    mParametersRef = graph.ImportBuffer(mParameters, mParametersAccess, kReadRest);
    mParametersAccess = kReadRest;
    mGradientsRef = graph.ImportBuffer(mGradients, mGradientsAccess, kWriteRest);
    mGradientsAccess = kWriteRest;
    mMatricesRef = {};
    if (!mLinearAlgebra)
    {
        return;
    }
    mMatricesRef = graph.ImportBuffer(mMatrices, mMatricesAccess, kReadRest);
    mMatricesAccess = kReadRest;
    if (mConvertPending)
    {
        mConvertPending = false;
        const rdg::BufferRef halfWeights = graph.ImportBuffer(mHalfWeights, mHalfWeightsAccess, kReadRest);
        mHalfWeightsAccess = kReadRest;
        AddWeightConversion(graph, halfWeights);
    }
}

void NeuralNetwork::AddWeightConversion(rdg::RenderGraph& graph, rdg::BufferRef halfWeights)
{
    const rdg::BufferRef matrices = mMatricesRef;
    const gpu::MlpBuffers buffers = mBuffers;
    graph.AddPass(
        mDesc.name + ".ConvertWeights", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(halfWeights, Access::SRVCompute);
            b.Write(matrices, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx)
        {
            rhi::MatrixConversion conversions[gpu::kMlpMatrices];
            for (u32 m = 0; m < gpu::kMlpMatrices; ++m)
            {
                conversions[m] = {.rows = gpu::kMlpMatrixRows[m],
                                  .columns = gpu::kMlpWidth,
                                  .src = ctx.Get(halfWeights),
                                  .srcOffset = u64(gpu::MlpMatrixOffset(m)) * 2,
                                  .srcLayout = rhi::MatrixLayout::RowMajor,
                                  .srcType = rhi::MatrixType::Float16,
                                  .srcStride = gpu::kMlpWidth * 2,
                                  .dst = ctx.Get(matrices),
                                  .dstOffset = gpu::MlpMatrixByteOffset(buffers, m),
                                  .dstLayout = rhi::MatrixLayout::MulOptimal,
                                  .dstType = rhi::MatrixType::Float16};
            }
            ctx.cmd.ConvertMatrices(conversions);
        });
}

void NeuralNetwork::AddOptimizerPasses(rdg::RenderGraph& graph, const NeuralTrainingSettings& settings, float samples)
{
    AIR_CHECK(mParametersRef.IsValid(), "{}: AddOptimizerPasses without BeginFrame()", mDesc.name);
    ++mStep;
    const rdg::BufferRef parameters = mParametersRef;
    const rdg::BufferRef gradients = mGradientsRef;
    const rdg::BufferRef moments = graph.ImportBuffer(mMoments, mMomentsAccess, kWriteRest);
    mMomentsAccess = kWriteRest;
    rdg::BufferRef rowMajor;
    rdg::BufferRef halfWeights;
    const gpu::MlpBuffers buffers = mBuffers;
    if (mLinearAlgebra)
    {
        rowMajor = graph.ImportBuffer(mGradientsRowMajor, mGradientsRowMajorAccess, kReadRest);
        mGradientsRowMajorAccess = kReadRest;
        halfWeights = graph.ImportBuffer(mHalfWeights, mHalfWeightsAccess, kReadRest);
        mHalfWeightsAccess = kReadRest;
        graph.AddPass(
            mDesc.name + ".ConvertGradients", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(gradients, Access::SRVCompute);
                b.Write(rowMajor, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx)
            {
                rhi::MatrixConversion conversions[gpu::kMlpMatrices];
                for (u32 m = 0; m < gpu::kMlpMatrices; ++m)
                {
                    conversions[m] = {.rows = gpu::kMlpMatrixRows[m],
                                      .columns = gpu::kMlpWidth,
                                      .src = ctx.Get(gradients),
                                      .srcOffset = gpu::MlpGradientByteOffset(buffers, m),
                                      .srcLayout = rhi::MatrixLayout::OuterProductOptimal,
                                      .srcType = rhi::MatrixType::Float32,
                                      .dst = ctx.Get(rowMajor),
                                      .dstOffset = u64(gpu::MlpMatrixOffset(m)) * 4,
                                      .dstLayout = rhi::MatrixLayout::RowMajor,
                                      .dstType = rhi::MatrixType::Float32,
                                      .dstStride = gpu::kMlpWidth * 4};
                }
                ctx.cmd.ConvertMatrices(conversions);
            });
    }

    gpu::MlpOptimizeConstants constants{};
    constants.parameterCount = mParameterCount;
    constants.matrixGradientWords = mMatrixGradientBytes / 4;
    constants.encodingGradients = mBuffers.encodingGradients;
    constants.flags = mLinearAlgebra ? gpu::kMlpOptimizeLinearAlgebra : 0u;
    constants.outputScale = mOutputScale;
    // The kernels summed kMlpLossScale times the loss gradients of about `samples` samples.
    constants.adam = MakeAdam(settings.learningRate, mStep, 1.0f / (std::max(samples, 1.0f) * gpu::kMlpLossScale),
                              settings.beta1, settings.beta2, settings.epsilon, settings.weightDecay);
    mOutputScale = 1.0f;
    const u32 threads = std::max(mParameterCount / 2, (mMatrixGradientBytes / 4 + 1) / 2);
    const rhi::PipelineHandle pipeline = mOptimizer;
    graph.AddPass(
        mDesc.name + ".Optimize", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Write(parameters, Access::UAVCompute);
            b.Write(moments, Access::UAVCompute);
            b.Write(gradients, Access::UAVCompute);
            if (rowMajor.IsValid())
            {
                b.Read(rowMajor, Access::SRVCompute);
                b.Write(halfWeights, Access::UAVCompute);
            }
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.parameters = ctx.UAV(parameters);
            constants.moments = ctx.UAV(moments);
            constants.gradients = ctx.UAV(gradients);
            if (rowMajor.IsValid())
            {
                constants.gradientsRowMajor = ctx.SRV(rowMajor);
                constants.halfWeights = ctx.UAV(halfWeights);
            }
            ctx.cmd.SetPipeline(pipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch((threads + 255) / 256, 1, 1);
        });

    if (mLinearAlgebra)
    {
        AddWeightConversion(graph, halfWeights);
    }
}

} // namespace air::renderer
