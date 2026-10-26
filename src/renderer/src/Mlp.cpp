#include "air/renderer/Mlp.h"

#include "air/core/Assert.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace air::renderer
{
namespace
{

using namespace air::hlsl;
using namespace air::gpu;

// The shader code becomes member functions; the hooks read `parameters`.
struct MlpKernel
{
    std::span<const float> parameters;

    float4 MlpLoadWeights4(uint index) const
    {
        return float4(parameters[index], parameters[index + 1], parameters[index + 2], parameters[index + 3]);
    }
    float2 MlpLoadFeatures(uint index) const { return float2(parameters[index], parameters[index + 1]); }

// NOLINTBEGIN(readability-identifier-naming)
#include "air/renderer/shader/Mlp.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

MlpKernel::MlpVector ToVector(std::span<const float> values)
{
    MlpKernel::MlpVector x;
    for (u32 i = 0; i < kMlpVector4s; ++i)
    {
        auto at = [&](u32 k) { return k < values.size() ? values[k] : 0.0f; };
        x.v[i] = float4(at(i * 4), at(i * 4 + 1), at(i * 4 + 2), at(i * 4 + 3));
    }
    return x;
}

} // namespace

std::vector<float> InitializeMlpParameters(u32 encodingParameters, u32 outputs, u32 seed)
{
    AIR_CHECK(outputs >= 1 && outputs <= kMlpOutputs, "an MLP has 1 .. {} outputs, not {}", kMlpOutputs, outputs);
    std::vector<float> parameters(kMlpWeights + encodingParameters, 0.0f);
    std::mt19937 rng(seed);
    for (u32 m = 0; m < kMlpMatrices; ++m)
    {
        const bool last = m + 1 == kMlpMatrices;
        const u32 rows = last ? outputs : kMlpWidth;
        const float limit = last ? std::sqrt(6.0f / float(kMlpWidth + outputs)) : std::sqrt(6.0f / float(kMlpWidth));
        std::uniform_real_distribution<float> uniform(-limit, limit);
        for (u32 i = 0; i < rows * kMlpWidth; ++i)
        {
            parameters[MlpMatrixOffset(m) + i] = uniform(rng);
        }
    }
    std::uniform_real_distribution<float> small(-1e-4f, 1e-4f);
    for (u32 i = kMlpWeights; i < parameters.size(); ++i)
    {
        parameters[i] = small(rng);
    }
    return parameters;
}

void EvaluateMlp(std::span<const float> parameters, std::span<const float> input, std::span<float> outputs)
{
    MlpKernel kernel{parameters}; // the shader functions are not const members
    const MlpKernel::MlpForward f = kernel.MlpEvaluate(ToVector(input));
    for (u32 i = 0; i < outputs.size() && i < kMlpOutputs; ++i)
    {
        outputs[i] = MlpComponent(f.output.v[i / 4], i % 4);
    }
}

void BackpropagateMlp(std::span<const float> parameters, std::span<const float> input,
                      std::span<const float> outputGradient, std::span<double> weightGradients,
                      std::span<float> inputGradient)
{
    AIR_CHECK(weightGradients.size() >= kMlpWeights, "BackpropagateMlp: {} weight gradients, need {}",
              weightGradients.size(), kMlpWeights);
    MlpKernel kernel{parameters};
    const MlpKernel::MlpForward f = kernel.MlpEvaluate(ToVector(input));
    const MlpKernel::MlpBackward b = kernel.MlpDifferentiate(f, ToVector(outputGradient));
    for (u32 m = 0; m < kMlpMatrices; ++m)
    {
        for (u32 r = 0; r < kMlpMatrixRows[m]; ++r)
        {
            const double g = kernel.MlpGet(b.outputGradient[m], r);
            if (g == 0.0)
            {
                continue;
            }
            double* row = &weightGradients[MlpMatrixOffset(m) + r * kMlpWidth];
            for (u32 c = 0; c < kMlpWidth; ++c)
            {
                row[c] += g * kernel.MlpGet(f.input[m], c);
            }
        }
    }
    for (u32 i = 0; i < inputGradient.size() && i < kMlpWidth; ++i)
    {
        inputGradient[i] = kernel.MlpGet(b.inputGradient, i);
    }
}

void EncodeHashGrid(std::span<const float> parameters, const HashGridData& grid, float3 p, std::span<float> inputs)
{
    MlpKernel kernel{parameters};
    MlpKernel::MlpVector x = kernel.MlpZero();
    kernel.MlpEncodeHashGrid(grid, p, x);
    for (u32 i = 0; i < inputs.size() && i < kHashInputs; ++i)
    {
        inputs[i] = kernel.MlpGet(x, i);
    }
}

void BackpropagateHashGrid(const HashGridData& grid, float3 p, std::span<const float> inputGradient,
                           std::span<double> parameterGradients)
{
    for (u32 level = 0; level < kHashLevels; ++level)
    {
        for (u32 corner = 0; corner < 8; ++corner)
        {
            uint entry = 0;
            float weight = 0.0f;
            MlpHashCorner(grid, p, level, corner, entry, weight);
            const u32 index = MlpHashParameter(grid, level, entry);
            for (u32 f = 0; f < kHashFeatures; ++f)
            {
                parameterGradients[index + f] += double(weight) * inputGradient[level * kHashFeatures + f];
            }
        }
    }
}

HashGridData MakeHashGrid(float3 boundsMin, float3 boundsMax, u32 tableSize, float coarsest, float finest,
                          u32 parameterOffset)
{
    AIR_CHECK(tableSize > 0 && (tableSize & (tableSize - 1)) == 0, "hash grid table size {} is not a power of two",
              tableSize);
    HashGridData grid{};
    const float3 extent = boundsMax - boundsMin;
    const float side = std::max(std::max(extent.x, extent.y), std::max(extent.z, 1e-6f));
    grid.origin = boundsMin;
    grid.scale = 1.0f / side;
    grid.tableSize = tableSize;
    grid.parameterOffset = parameterOffset;
    const double growth = std::exp(std::log(double(finest) / double(coarsest)) / double(kHashLevels - 1));
    for (u32 level = 0; level < kHashLevels; ++level)
    {
        const float resolution = float(std::max(1.0, std::floor(double(coarsest) * std::pow(growth, double(level)))));
        float4& r = grid.resolution[level / 4];
        (level % 4 == 0 ? r.x : level % 4 == 1 ? r.y : level % 4 == 2 ? r.z : r.w) = resolution;
    }
    return grid;
}

AdamData MakeAdam(float learningRate, u32 step, float gradientScale, float beta1, float beta2, float epsilon,
                  float weightDecay)
{
    AdamData adam{};
    adam.learningRate = learningRate;
    adam.beta1 = beta1;
    adam.beta2 = beta2;
    adam.epsilon = epsilon;
    adam.correction1 = float(1.0 / (1.0 - std::pow(double(beta1), double(std::max(step, 1u)))));
    adam.correction2 = float(1.0 / (1.0 - std::pow(double(beta2), double(std::max(step, 1u)))));
    adam.weightDecay = weightDecay;
    adam.gradientScale = gradientScale;
    return adam;
}

void AdamStep(const AdamData& adam, std::span<float> parameters, std::span<float> m, std::span<float> v,
              std::span<const double> gradients)
{
    for (size_t i = 0; i < parameters.size(); ++i)
    {
        const bool weight = i < kMlpWeights;
        if (!weight && gradients[i] == 0.0)
        {
            continue; // an encoding entry no sample touched keeps its moments
        }
        parameters[i] = AdamUpdate(adam, parameters[i], float(gradients[i]), weight, m[i], v[i]);
    }
}

} // namespace air::renderer
