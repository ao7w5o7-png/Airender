// The multilayer perceptron of MlpShared.h in float32: evaluation and backpropagation, written in the
// HLSL / C++ subset so the CPU (tests, reference) and GPUs without linear algebra run the same code.
//
// The including code defines, before including this file:
//   float4 MlpLoadWeights4(uint index)   parameters index .. index + 3 (index a multiple of 4)
//   float2 MlpLoadFeatures(uint index)   parameters index, index + 1 (a hash grid entry)
#ifndef AIR_RENDERER_MLP_HLSLI
#define AIR_RENDERER_MLP_HLSLI

#include "air/renderer/shader/MlpShared.h"

struct MlpVector
{
    float4 v[kMlpVector4s];
};

MlpVector MlpZero()
{
    MlpVector x;
    AIR_UNROLL
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        x.v[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return x;
}

float MlpGet(MlpVector x, uint i)
{
    return MlpComponent(x.v[i / 4u], i % 4u);
}

// The inputs of position p from the hash grid: x.v[0 .. kHashInputs / 4 - 1].
void MlpEncodeHashGrid(HashGridData g, float3 p, AIR_INOUT(MlpVector) x)
{
    AIR_UNROLL
    for (uint pair = 0u; pair < kHashLevels / 2u; ++pair)
    {
        float2 features[2];
        AIR_UNROLL
        for (uint k = 0u; k < 2u; ++k)
        {
            uint level = pair * 2u + k;
            features[k] = float2(0.0f, 0.0f);
            for (uint corner = 0u; corner < 8u; ++corner)
            {
                uint entry = 0u;
                float weight = 0.0f;
                MlpHashCorner(g, p, level, corner, entry, weight);
                features[k] = features[k] + MlpLoadFeatures(MlpHashParameter(g, level, entry)) * weight;
            }
        }
        x.v[pair] = float4(features[0].x, features[0].y, features[1].x, features[1].y);
    }
}

// Row r of matrix m times x.
float MlpRowDot(uint m, uint r, MlpVector x)
{
    uint base = MlpMatrixOffset(m) + r * kMlpWidth;
    float sum = 0.0f;
    AIR_UNROLL
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        sum += dot(MlpLoadWeights4(base + i * 4u), x.v[i]);
    }
    return sum;
}

// W_m x: kMlpMatrixRows[m] outputs, the rest zero.
MlpVector MlpMultiply(uint m, MlpVector x)
{
    MlpVector y = MlpZero();
    uint rows = kMlpMatrixRows[m];
    AIR_UNROLL
    for (uint r4 = 0u; r4 < kMlpVector4s; ++r4)
    {
        if (r4 * 4u < rows)
        {
            y.v[r4] = float4(MlpRowDot(m, r4 * 4u, x), MlpRowDot(m, r4 * 4u + 1u, x), MlpRowDot(m, r4 * 4u + 2u, x),
                             MlpRowDot(m, r4 * 4u + 3u, x));
        }
    }
    return y;
}

// W_m^T y: what an output gradient y of matrix m means for its inputs.
MlpVector MlpMultiplyTransposed(uint m, MlpVector y)
{
    MlpVector x = MlpZero();
    uint rows = kMlpMatrixRows[m];
    for (uint r = 0u; r < rows; ++r)
    {
        float yr = MlpGet(y, r);
        if (yr != 0.0f)
        {
            uint base = MlpMatrixOffset(m) + r * kMlpWidth;
            AIR_UNROLL
            for (uint i = 0u; i < kMlpVector4s; ++i)
            {
                x.v[i] = x.v[i] + MlpLoadWeights4(base + i * 4u) * yr;
            }
        }
    }
    return x;
}

MlpVector MlpRelu(MlpVector z)
{
    MlpVector a;
    AIR_UNROLL
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        float4 v = z.v[i];
        a.v[i] = float4(max(v.x, 0.0f), max(v.y, 0.0f), max(v.z, 0.0f), max(v.w, 0.0f));
    }
    return a;
}

// g where the ReLU that produced `activation` passed its input on, 0 elsewhere.
MlpVector MlpReluGradient(MlpVector g, MlpVector activation)
{
    MlpVector d;
    AIR_UNROLL
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        float4 a = activation.v[i];
        float4 v = g.v[i];
        d.v[i] = float4(a.x > 0.0f ? v.x : 0.0f, a.y > 0.0f ? v.y : 0.0f, a.z > 0.0f ? v.z : 0.0f,
                        a.w > 0.0f ? v.w : 0.0f);
    }
    return d;
}

// A forward pass, keeping what backpropagation needs: the input of every matrix (the encoded input,
// then the hidden layers' activations) and the output.
struct MlpForward
{
    MlpVector input[kMlpMatrices];
    MlpVector output;
};

MlpForward MlpEvaluate(MlpVector x)
{
    MlpForward f;
    f.input[0] = x;
    AIR_UNROLL
    for (uint m = 0u; m + 1u < kMlpMatrices; ++m)
    {
        f.input[m + 1u] = MlpRelu(MlpMultiply(m, f.input[m]));
    }
    f.output = MlpMultiply(kMlpMatrices - 1u, f.input[kMlpMatrices - 1u]);
    return f;
}

// Backpropagation of the loss gradient dy (with respect to the outputs): the gradient with respect to
// every matrix's outputs (the weight gradient of matrix m is outputGradient[m] (outer) input[m]) and
// with respect to the network's input (for a trainable encoding).
struct MlpBackward
{
    MlpVector outputGradient[kMlpMatrices];
    MlpVector inputGradient;
};

MlpBackward MlpDifferentiate(MlpForward f, MlpVector dy)
{
    MlpBackward b;
    b.outputGradient[kMlpMatrices - 1u] = dy;
    AIR_UNROLL
    for (uint k = 1u; k < kMlpMatrices; ++k)
    {
        uint m = kMlpMatrices - 1u - k; // matrix m feeds the ReLU whose output is input[m + 1]
        b.outputGradient[m] = MlpReluGradient(MlpMultiplyTransposed(m + 1u, b.outputGradient[m + 1u]), f.input[m + 1u]);
    }
    b.inputGradient = MlpMultiplyTransposed(0u, b.outputGradient[0]);
    return b;
}

#endif
