#pragma once

// CPU twin of the renderer's neural networks (shader/MlpShared.h, shader/Mlp.hlsli): the same
// float32 code, for tests (backpropagation against finite differences, GPU kernels against the CPU)
// and for CPU renders with a cache the GPU trained. docs/architecture.md §8.11.

#include "air/core/Types.h"
#include "air/renderer/shader/MlpShared.h"

#include <span>
#include <vector>

namespace air::renderer
{

// Parameters of a network: the MLP's weights, then `encodingParameters` of its encoding. Hidden
// layers start He-uniform (ReLU), the output layer Glorot-uniform, the hash grid uniform in +-1e-4
// (Mueller et al. 2022). Rows of the output layer beyond `outputs` start at zero: their gradients are
// zero, so they stay there.
std::vector<float> InitializeMlpParameters(u32 encodingParameters, u32 outputs, u32 seed);

// The network's outputs for an encoded input.
void EvaluateMlp(std::span<const float> parameters, std::span<const float> input, std::span<float> outputs);

// Backpropagates one sample: adds the gradient of a loss whose gradient with respect to the outputs
// is `outputGradient` to `weightGradients` (kMlpWeights values) and returns the gradient with respect
// to the encoded input in `inputGradient` (kMlpWidth values).
void BackpropagateMlp(std::span<const float> parameters, std::span<const float> input,
                      std::span<const float> outputGradient, std::span<double> weightGradients,
                      std::span<float> inputGradient);

// The hash grid's inputs (kHashInputs values) at position p.
void EncodeHashGrid(std::span<const float> parameters, const gpu::HashGridData& grid, gpu::float3 p,
                    std::span<float> inputs);
// Adds the gradient of the hash grid's parameters, given the gradient with respect to its inputs.
void BackpropagateHashGrid(const gpu::HashGridData& grid, gpu::float3 p, std::span<const float> inputGradient,
                           std::span<double> parameterGradients);

// A hash grid of `tableSize` entries per level over the box [boundsMin, boundsMax] (its longest side),
// levels from `coarsest` to `finest` cells across, growing geometrically.
gpu::HashGridData MakeHashGrid(gpu::float3 boundsMin, gpu::float3 boundsMax, u32 tableSize, float coarsest,
                               float finest, u32 parameterOffset);

// Adam's constants for `step` (1, 2, ...).
gpu::AdamData MakeAdam(float learningRate, u32 step, float gradientScale, float beta1 = 0.9f, float beta2 = 0.99f,
                       float epsilon = 1e-8f, float weightDecay = 1e-6f);

// One optimizer step over every parameter, the summed gradients in `gradients`. Like the GPU, it skips
// encoding parameters whose gradient is exactly zero (no sample touched them).
void AdamStep(const gpu::AdamData& adam, std::span<float> parameters, std::span<float> m, std::span<float> v,
              std::span<const double> gradients);

} // namespace air::renderer
