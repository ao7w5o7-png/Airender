#pragma once

// A neural network trained on the GPU inside the frame (shader/MlpShared.h; docs/architecture.md
// §8.11): its parameters, Adam's moments and the gradients live in buffers across frames, and each
// training step is a few render graph passes:
//
//   <the application's kernels>   accumulate gradients into GetBuffers().gradients
//   Mlp.ConvertGradients          linear algebra: the MLP's gradients to row major
//   Mlp.Optimize                  Adam on every parameter (MlpOptimize.hlsl), clears the gradients
//   Mlp.ConvertWeights            linear algebra: the new float16 weights to the MulOptimal layout
//
// With DeviceCaps::linearAlgebra the application's kernels multiply on the matrix units
// (shader/MlpLinAlg.hlsli); without, they run shader/Mlp.hlsli in float32.

#include "air/core/Json.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/shader/MlpShared.h"
#include "air/rhi/RHIDevice.h"

#include <filesystem>
#include <string>
#include <vector>

namespace air::renderer
{

struct NeuralNetworkDesc
{
    u32 encodingParameters = 0; // trainable parameters of the input encoding, after the MLP's
    u32 outputs = 3;            // output rows in use (at most kMlpOutputs)
    std::string name = "Mlp";   // prefix of the passes' and buffers' debug names
};

struct NeuralTrainingSettings
{
    float learningRate = 1e-2f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-8f;
    float weightDecay = 1e-6f;
};

class NeuralNetwork
{
public:
    NeuralNetwork(rhi::Device& device, NeuralNetworkDesc desc);
    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    // Loads the optimizer (MlpOptimize.cs from the renderer's shaders, <shader root>/renderer) and
    // creates the buffers with freshly initialized parameters. linearAlgebra: use the matrix units
    // where the device has them. False without the optimizer.
    bool Initialize(const std::filesystem::path& rendererShaderDir, bool linearAlgebra, u32 seed);
    bool UsesLinearAlgebra() const { return mLinearAlgebra; }

    // Starts over from freshly initialized parameters (renderer::InitializeMlpParameters).
    void Reset(u32 seed);
    // Starts from these parameters (the MLP's weights, then the encoding's).
    void Upload(const std::vector<float>& parameters);

    // Imports the buffers into this frame's graph; after a reset the first pass converts the uploaded
    // weights. Call before adding passes that use the network.
    void BeginFrame(rdg::RenderGraph& graph);
    // Bindless indices and offsets for kernels (valid after Initialize).
    gpu::MlpBuffers GetBuffers() const { return mBuffers; }
    // This frame's buffers, for passes to declare: parameters and matrices are read, gradients written.
    rdg::BufferRef Parameters() const { return mParametersRef; }
    rdg::BufferRef Matrices() const { return mMatricesRef; } // invalid without linear algebra
    rdg::BufferRef Gradients() const { return mGradientsRef; }

    // One optimizer step over the gradients the passes since the last one accumulated: kMlpLossScale
    // times the loss gradients of about `samples` samples (both divided out).
    void AddOptimizerPasses(rdg::RenderGraph& graph, const NeuralTrainingSettings& settings, float samples);
    // The output layer is multiplied by `factor` at the next step (the targets were rescaled).
    void ScaleOutputs(float factor) { mOutputScale *= factor; }
    u32 GetStep() const { return mStep; }
    u32 GetParameterCount() const { return mParameterCount; }

    // For checks between frames, waiting for the GPU: the parameters, and the MLP's weight gradients
    // summed since the last optimizer step (row major, the kernels' kMlpLossScale included).
    std::vector<float> ReadParameters();
    std::vector<float> ReadWeightGradients();

private:
    void CreateBuffers(const std::vector<float>& parameters);
    void ReleaseBuffers();
    // Linear algebra: the float16 row-major weights to this frame's MulOptimal matrices.
    void AddWeightConversion(rdg::RenderGraph& graph, rdg::BufferRef halfWeights);

    rhi::Device& mDevice;
    NeuralNetworkDesc mDesc;
    bool mLinearAlgebra = false;
    rhi::PipelineHandle mOptimizer;
    u32 mParameterCount = 0;

    rhi::BufferHandle mParameters;        // float32
    rhi::BufferHandle mMoments;           // float32 m, v
    rhi::BufferHandle mGradients;         // see MlpBuffers
    rhi::BufferHandle mGradientsRowMajor; // linear algebra: float32
    rhi::BufferHandle mHalfWeights;       // linear algebra: float16 row major
    rhi::BufferHandle mMatrices;          // linear algebra: float16 MulOptimal
    rhi::Access mParametersAccess = rhi::Access::None;
    rhi::Access mMomentsAccess = rhi::Access::None;
    rhi::Access mGradientsAccess = rhi::Access::None;
    rhi::Access mGradientsRowMajorAccess = rhi::Access::None;
    rhi::Access mHalfWeightsAccess = rhi::Access::None;
    rhi::Access mMatricesAccess = rhi::Access::None;
    u32 mMatrixGradientBytes = 0; // the MLP's part of mGradients
    gpu::MlpBuffers mBuffers{};

    rdg::BufferRef mParametersRef;
    rdg::BufferRef mMatricesRef;
    rdg::BufferRef mGradientsRef;
    bool mConvertPending = false; // uploaded float16 weights still need their MulOptimal conversion
    u32 mStep = 0;
    float mOutputScale = 1.0f;
};

// The GPU's network kernels against the CPU (aircli mlp-check): for `samples` random inputs and loss
// gradients, the outputs, the input gradients and the summed weight gradients that the float32 kernels
// and, where the device has them, the linear algebra ones compute, compared with renderer/Mlp.h's.
// JSON: per kernel kind the largest error relative to each quantity's largest magnitude, and
// "passed" (float32 within 1e-4, float16 linear algebra within 2e-2).
json::Value CheckNeuralNetwork(rhi::Device& device, const std::filesystem::path& rendererShaderDir, u32 samples,
                               u32 seed);

} // namespace air::renderer
