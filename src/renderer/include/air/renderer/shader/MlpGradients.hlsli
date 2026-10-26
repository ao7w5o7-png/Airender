// Summing the gradients of a training step's samples on the GPU (see MlpBuffers): 64-bit fixed point
// atomics, which unlike float atomics give the same sums in any order. GPU only.
//
// Include after Mlp.hlsli (the including code defines its hooks).
#ifndef AIR_RENDERER_MLP_GRADIENTS_HLSLI
#define AIR_RENDERER_MLP_GRADIENTS_HLSLI

#include "air/renderer/shader/Mlp.hlsli"

void MlpAddFixedPoint(RWByteAddressBuffer buffer, uint offset, float value)
{
    if (value != 0.0f)
    {
        buffer.InterlockedAdd64(offset, int64_t(round(value * kGradientFixedPoint)));
    }
}

// The hash grid's parameter gradients of one sample at p, from the gradient with respect to the
// network's input (x.v[0 .. kHashInputs / 4 - 1] are the grid's).
void MlpScatterHashGrid(MlpBuffers b, HashGridData g, float3 p, MlpVector inputGradient)
{
    RWByteAddressBuffer gradients = ResourceDescriptorHeap[b.gradients];
    for (uint level = 0u; level < kHashLevels; ++level)
    {
        float4 pair = inputGradient.v[level / 2u];
        float2 d = (level & 1u) != 0u ? float2(pair.z, pair.w) : float2(pair.x, pair.y);
        if (d.x == 0.0f && d.y == 0.0f)
        {
            continue;
        }
        for (uint corner = 0u; corner < 8u; ++corner)
        {
            uint entry = 0u;
            float weight = 0.0f;
            MlpHashCorner(g, p, level, corner, entry, weight);
            uint offset = b.encodingGradients + (MlpHashParameter(g, level, entry) - g.parameterOffset) * 8u;
            MlpAddFixedPoint(gradients, offset, d.x * weight);
            MlpAddFixedPoint(gradients, offset + 8u, d.y * weight);
        }
    }
}

// The MLP's weight gradients of a wave's samples (float32 path): every lane of the wave calls this,
// samples that do not train with zero gradients. Each weight's sum over the wave is added once.
void MlpAccumulateWeightGradients(MlpBuffers b, MlpForward f, MlpBackward d)
{
    RWByteAddressBuffer gradients = ResourceDescriptorHeap[b.gradients];
    for (uint m = 0u; m < kMlpMatrices; ++m)
    {
        uint base = MlpGradientByteOffset(b, m);
        for (uint r = 0u; r < kMlpMatrixRows[m]; ++r)
        {
            float g = MlpGet(d.outputGradient[m], r);
            if (!WaveActiveAnyTrue(g != 0.0f))
            {
                continue; // no lane's ReLU passed this neuron
            }
            for (uint c = 0u; c < kMlpVector4s; ++c)
            {
                float4 sum = WaveActiveSum(f.input[m].v[c] * g);
                if (WaveIsFirstLane())
                {
                    uint offset = base + (r * kMlpWidth + c * 4u) * 8u;
                    MlpAddFixedPoint(gradients, offset, sum.x);
                    MlpAddFixedPoint(gradients, offset + 8u, sum.y);
                    MlpAddFixedPoint(gradients, offset + 16u, sum.z);
                    MlpAddFixedPoint(gradients, offset + 24u, sum.w);
                }
            }
        }
    }
}

#endif
