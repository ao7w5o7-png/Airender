// Small neural networks evaluated and trained inside the renderer's passes, shared by C++ and HLSL
// (see HlslCompat.h). docs/architecture.md §8.11.
//
// The network is a multilayer perceptron of kMlpWidth inputs, kMlpHiddenLayers hidden layers of
// kMlpWidth neurons (ReLU) and kMlpOutputs linear outputs, of which an application uses the first
// few. There are no biases: an input held at 1 gives the first layer one. Its weights are
// kMlpMatrices row-major matrices of kMlpWidth columns, one after the other: kMlpWidth rows each,
// kMlpOutputs for the last. Inputs come from encodings: a multiresolution hash grid for positions
// (Mueller et al. 2022) and spherical harmonics for directions.
//
// Parameters (float32, "master" copy): the MLP's weights, then the encoding's (the hash grid: level
// after level, tableSize entries of kHashFeatures features). Mlp.hlsli evaluates and differentiates
// the network in float32 (CPU, and GPUs without linear algebra); MlpLinAlg.hlsli does the same with
// Shader Model 6.10 thread-scope matrices in float16 on the GPU's matrix units.
#ifndef AIR_RENDERER_MLP_SHARED_H
#define AIR_RENDERER_MLP_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kMlpWidth = 64;
AIR_CONST uint kMlpHiddenLayers = 2;
AIR_CONST uint kMlpOutputs = 16;
AIR_CONST uint kMlpMatrices = kMlpHiddenLayers + 1;
AIR_CONST uint kMlpVector4s = kMlpWidth / 4; // float4s per activation vector
AIR_CONST uint kMlpWeights = kMlpHiddenLayers * kMlpWidth * kMlpWidth + kMlpOutputs * kMlpWidth;

// Multiresolution hash grid: kHashLevels levels of kHashFeatures features (kHashLevels *
// kHashFeatures network inputs).
AIR_CONST uint kHashLevels = 16;
AIR_CONST uint kHashFeatures = 2;
AIR_CONST uint kHashInputs = kHashLevels * kHashFeatures;

// Gradients summed with 64-bit integer atomics (deterministic, unlike float atomics) are fixed point
// with this scale: 2^-32 resolution, up to 2^31 in magnitude.
AIR_CONST float kGradientFixedPoint = 4294967296.0f;

struct HashGridData
{
    float4 resolution[kHashLevels / 4]; // cells across the grid per level (whole numbers)
    float3 origin;                      // world position of the grid's corner
    float scale;                        // 1 / the grid's extent (world units): positions map to [0, 1]^3
    uint tableSize;                     // entries per level, a power of two
    uint parameterOffset;               // of level 0, entry 0, in the parameters (floats)
    uint pad0;
    uint pad1;
};

// Adam (Kingma and Ba 2015) as the optimizer applies it to every parameter of a step.
struct AdamData
{
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float correction1;   // 1 / (1 - beta1^step)
    float correction2;   // 1 / (1 - beta2^step)
    float weightDecay;   // L2 regularization of the MLP's weights (not the encoding's)
    float gradientScale; // multiplies the summed gradients (1 / samples)
};

// Where kernels find a network on the GPU (renderer::NeuralNetwork::GetBuffers): bindless indices
// and byte offsets. With linear algebra the kernels multiply with the float16 matrices and accumulate
// the MLP's gradients as float32 matrices in the device's OuterProductOptimal layout; without, they
// read the float32 parameters and sum the gradients as 64-bit fixed point (kGradientFixedPoint), row
// major. The encoding's gradients are always fixed point, one int64 per parameter.
struct MlpBuffers
{
    uint parameters;    // ByteAddressBuffer: float32 parameters
    uint matrices;      // ByteAddressBuffer: the MLP's float16 MulOptimal matrices (linear algebra)
    uint gradients;     // RWByteAddressBuffer
    uint linearAlgebra; // 1: the kernels use `matrices`
    uint matrixOffset0; // bytes, of matrix 0 in `matrices`
    uint matrixOffset1;
    uint matrixOffset2;
    uint gradientOffset0; // bytes, of matrix 0's gradients in `gradients`
    uint gradientOffset1;
    uint gradientOffset2;
    uint encodingGradients; // bytes, of the first encoding parameter's gradient in `gradients`
    uint pad0;
};

AIR_CONST uint kMlpOptimizeLinearAlgebra = 1u; // MlpOptimizeConstants::flags

// MlpOptimize.hlsl: one Adam step of every parameter (two per thread), then the gradients start over.
struct MlpOptimizeConstants
{
    uint parameters;          // RWByteAddressBuffer: float32
    uint moments;             // RWByteAddressBuffer: float32 m, v per parameter
    uint gradients;           // RWByteAddressBuffer (MlpBuffers::gradients)
    uint gradientsRowMajor;   // ByteAddressBuffer: the MLP's gradients converted to row major (linear algebra)
    uint halfWeights;         // RWByteAddressBuffer: float16 row-major copy of the MLP's weights (linear algebra)
    uint parameterCount;      // even
    uint matrixGradientWords; // 32-bit words of the MLP's gradients in `gradients` (linear algebra)
    uint encodingGradients;   // bytes, as MlpBuffers
    uint flags;               // kMlpOptimize*
    float outputScale;        // multiplies the output layer's weights and moments first (1: none)
    uint pad0;
    uint pad1;
    AdamData adam;
};

// MlpCheck*.hlsl (renderer::CheckNeuralNetwork, aircli mlp-check): per sample, the network's outputs
// and its gradients for a given loss gradient, by the GPU's kernels for comparison with the CPU's.
struct MlpCheckConstants
{
    MlpBuffers network;
    uint inputs;          // ByteAddressBuffer: kMlpWidth float32 per sample
    uint outputGradients; // ByteAddressBuffer: kMlpOutputs float32 per sample, the loss gradient
    uint outputs;         // RWByteAddressBuffer: kMlpOutputs float32 per sample
    uint inputGradients;  // RWByteAddressBuffer: kMlpWidth float32 per sample
    uint samples;
    uint pad0;
    uint pad1;
    uint pad2;
};

#ifdef __cplusplus
static_assert(sizeof(HashGridData) == 96);
static_assert(sizeof(AdamData) == 32);
static_assert(sizeof(MlpBuffers) == 48);
static_assert(sizeof(MlpCheckConstants) == 80);
static_assert(sizeof(MlpOptimizeConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(kMlpMatrices == 3, "MlpBuffers has an offset per matrix");
#endif

AIR_INLINE uint MlpMatrixByteOffset(MlpBuffers b, uint m)
{
    return m == 0u ? b.matrixOffset0 : m == 1u ? b.matrixOffset1 : b.matrixOffset2;
}

AIR_INLINE uint MlpGradientByteOffset(MlpBuffers b, uint m)
{
    return m == 0u ? b.gradientOffset0 : m == 1u ? b.gradientOffset1 : b.gradientOffset2;
}

AIR_CONST uint kMlpMatrixRows[kMlpMatrices] = {kMlpWidth, kMlpWidth, kMlpOutputs};

// Offset (floats) of matrix m's first weight.
AIR_INLINE uint MlpMatrixOffset(uint m)
{
    return m * kMlpWidth * kMlpWidth;
}

AIR_INLINE float MlpComponent(float4 v, uint i)
{
    return i == 0u ? v.x : i == 1u ? v.y : i == 2u ? v.z : v.w;
}

AIR_INLINE float MlpHashResolution(HashGridData g, uint level)
{
    return MlpComponent(g.resolution[level / 4u], level % 4u);
}

// --- Encodings -------------------------------------------------------------------------------------

// Real spherical harmonics up to degree 3 of a unit direction (16 values).
AIR_INLINE void MlpSphericalHarmonics(float3 d, AIR_OUT(float4) sh0, AIR_OUT(float4) sh1, AIR_OUT(float4) sh2,
                                      AIR_OUT(float4) sh3)
{
    float x = d.x;
    float y = d.y;
    float z = d.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    sh0 = float4(0.28209479177387814f, -0.48860251190291987f * y, 0.48860251190291987f * z, -0.48860251190291987f * x);
    sh1 = float4(1.0925484305920792f * x * y, -1.0925484305920792f * y * z,
                 0.94617469575755997f * zz - 0.31539156525251999f, -1.0925484305920792f * x * z);
    sh2 = float4(0.54627421529603959f * (xx - yy), 0.59004358992664352f * y * (-3.0f * xx + yy),
                 2.8906114426405538f * x * y * z, 0.45704579946446572f * y * (1.0f - 5.0f * zz));
    sh3 = float4(0.3731763325901154f * z * (5.0f * zz - 3.0f), 0.45704579946446572f * x * (1.0f - 5.0f * zz),
                 1.4453057213202769f * z * (xx - yy), 0.59004358992664352f * x * (-xx + 3.0f * yy));
}

// One of the 8 grid vertices around position p at `level`: the index of its entry (relative to the
// level's first) and its trilinear weight. Coarse levels whose vertices fit the table map them one to
// one; finer ones hash them (Teschner et al. 2003 primes, as Mueller et al.).
AIR_INLINE void MlpHashCorner(HashGridData g, float3 p, uint level, uint corner, AIR_OUT(uint) entry,
                              AIR_OUT(float) weight)
{
    float resolution = MlpHashResolution(g, level);
    float3 q = (p - g.origin) * g.scale;
    float qx = clamp(q.x, 0.0f, 1.0f) * resolution;
    float qy = clamp(q.y, 0.0f, 1.0f) * resolution;
    float qz = clamp(q.z, 0.0f, 1.0f) * resolution;
    float bx = min(floor(qx), resolution - 1.0f);
    float by = min(floor(qy), resolution - 1.0f);
    float bz = min(floor(qz), resolution - 1.0f);
    float fx = qx - bx;
    float fy = qy - by;
    float fz = qz - bz;
    uint ox = corner & 1u;
    uint oy = (corner >> 1u) & 1u;
    uint oz = (corner >> 2u) & 1u;
    weight = (ox != 0u ? fx : 1.0f - fx) * (oy != 0u ? fy : 1.0f - fy) * (oz != 0u ? fz : 1.0f - fz);
    uint x = uint(bx) + ox;
    uint y = uint(by) + oy;
    uint z = uint(bz) + oz;
    uint side = uint(resolution) + 1u;
    if (float(side) * float(side) * float(side) <= float(g.tableSize))
    {
        entry = x + side * (y + side * z);
    }
    else
    {
        entry = (x ^ (y * 2654435761u) ^ (z * 805459861u)) & (g.tableSize - 1u);
    }
}

// Offset (floats) of an entry's first feature in the parameters.
AIR_INLINE uint MlpHashParameter(HashGridData g, uint level, uint entry)
{
    return g.parameterOffset + (level * g.tableSize + entry) * kHashFeatures;
}

// --- Training ----------------------------------------------------------------------------------------

AIR_CONST float kMlpLossEpsilon = 0.01f; // of the relative loss, in the targets' units
// Loss gradients are clamped to this per output and sample only to keep float16 finite: a tighter
// clamp biases heavy-tailed targets low (a cache learns their mean, which the rare bright samples
// carry; clamping at 64 made the door scene's cache 28% dark).
AIR_CONST float kMlpMaxLossGradient = 65536.0f;
// Training kernels backpropagate this times the loss gradient, which keeps float16 activations'
// gradients in range (mixed precision loss scaling); the optimizer divides it out.
AIR_CONST float kMlpLossScale = 1.0f / 64.0f;

// Gradient of the relative L2 loss sum_c (y_c - t_c)^2 / (L(y)^2 + epsilon) with respect to the
// prediction y, the luminance L(y) held constant (Mueller et al. 2021): errors count relative to how
// bright the prediction is, so dark and bright regions train alike.
AIR_INLINE float3 MlpLossGradient(float3 prediction, float3 target)
{
    float luminance = max(0.2126f * prediction.x + 0.7152f * prediction.y + 0.0722f * prediction.z, 0.0f);
    float3 g = (prediction - target) * (2.0f / (luminance * luminance + kMlpLossEpsilon));
    return float3(clamp(g.x, -kMlpMaxLossGradient, kMlpMaxLossGradient),
                  clamp(g.y, -kMlpMaxLossGradient, kMlpMaxLossGradient),
                  clamp(g.z, -kMlpMaxLossGradient, kMlpMaxLossGradient));
}

AIR_INLINE float MlpLoss(float3 prediction, float3 target)
{
    float luminance = max(0.2126f * prediction.x + 0.7152f * prediction.y + 0.0722f * prediction.z, 0.0f);
    float3 e = prediction - target;
    return dot(e, e) / (luminance * luminance + kMlpLossEpsilon);
}

// One Adam step of a parameter with summed gradient g; m and v are its moment estimates.
AIR_INLINE float AdamUpdate(AdamData a, float parameter, float g, bool decay, AIR_INOUT(float) m, AIR_INOUT(float) v)
{
    g = g * a.gradientScale + (decay ? a.weightDecay * parameter : 0.0f);
    m = a.beta1 * m + (1.0f - a.beta1) * g;
    v = a.beta2 * v + (1.0f - a.beta2) * g * g;
    float mHat = m * a.correction1;
    float vHat = v * a.correction2;
    return parameter - a.learningRate * mHat / (sqrt(vHat) + a.epsilon);
}

#ifdef __cplusplus
} // namespace air::gpu
#endif

#endif
