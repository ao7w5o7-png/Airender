// One Adam step of a neural network's parameters (renderer::NeuralNetwork, MlpShared.h): two
// parameters per thread. Reads the step's summed gradients and clears them for the next step; with
// linear algebra also writes the float16 copy of the MLP's weights that the next conversion turns
// into the MulOptimal matrices. Encoding parameters no sample touched (gradient exactly 0) keep their
// value and moments (sparse updates, as Mueller et al. 2022).

#include "air/renderer/shader/MlpShared.h"

[[vk::push_constant]] ConstantBuffer<MlpOptimizeConstants> g : register(b0);

float FixedPointToFloat(uint2 bits)
{
    int64_t value = int64_t((uint64_t(bits.y) << 32) | uint64_t(bits.x));
    return float(value) * (1.0f / kGradientFixedPoint);
}

[numthreads(256, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    RWByteAddressBuffer parameters = ResourceDescriptorHeap[g.parameters];
    RWByteAddressBuffer moments = ResourceDescriptorHeap[g.moments];
    RWByteAddressBuffer gradients = ResourceDescriptorHeap[g.gradients];
    bool linearAlgebra = (g.flags & kMlpOptimizeLinearAlgebra) != 0u;

    // With linear algebra the MLP's gradients are matrices in the device's layout: clear them as words.
    uint word = id.x * 2u;
    if (linearAlgebra && word < g.matrixGradientWords)
    {
        gradients.Store2(word * 4u, uint2(0u, 0u));
    }

    uint first = id.x * 2u;
    if (first >= g.parameterCount)
    {
        return;
    }
    if (first >= kMlpWeights)
    {
        // A hash grid entry (both its features): one no sample touched still has a zero gradient and
        // keeps its value and moments, without a single load beyond the gradient's.
        uint offset = g.encodingGradients + (first - kMlpWeights) * 8u;
        uint4 bits = gradients.Load4(offset);
        if (all(bits == 0u))
        {
            return;
        }
        gradients.Store4(offset, uint4(0u, 0u, 0u, 0u));
        float2 gradient = float2(FixedPointToFloat(bits.xy), FixedPointToFloat(bits.zw));
        float2 p = asfloat(parameters.Load2(first * 4u));
        float4 mv = asfloat(moments.Load4(first * 8u));
        if (gradient.x != 0.0f)
        {
            p.x = AdamUpdate(g.adam, p.x, gradient.x, false, mv.x, mv.y);
        }
        if (gradient.y != 0.0f)
        {
            p.y = AdamUpdate(g.adam, p.y, gradient.y, false, mv.z, mv.w);
        }
        parameters.Store2(first * 4u, asuint(p));
        moments.Store4(first * 8u, asuint(mv));
        return;
    }

    // The MLP's weights: every one changes.
    float updated[2];
    for (uint k = 0u; k < 2u; ++k)
    {
        uint i = first + k;
        float gradient = 0.0f;
        if (linearAlgebra)
        {
            ByteAddressBuffer rowMajor = ResourceDescriptorHeap[g.gradientsRowMajor];
            gradient = asfloat(rowMajor.Load(i * 4u));
        }
        else
        {
            gradient = FixedPointToFloat(gradients.Load2(i * 8u));
            gradients.Store2(i * 8u, uint2(0u, 0u));
        }
        float p = asfloat(parameters.Load(i * 4u));
        float2 mv = asfloat(moments.Load2(i * 8u));
        if (i >= MlpMatrixOffset(kMlpMatrices - 1u) && g.outputScale != 1.0f)
        {
            // The targets' scale changed: so does the output layer, moments included.
            p *= g.outputScale;
            mv *= float2(g.outputScale, g.outputScale * g.outputScale);
        }
        p = AdamUpdate(g.adam, p, gradient, true, mv.x, mv.y);
        parameters.Store(i * 4u, asuint(p));
        moments.Store2(i * 8u, asuint(mv));
        updated[k] = p;
    }
    if (linearAlgebra)
    {
        RWByteAddressBuffer halfWeights = ResourceDescriptorHeap[g.halfWeights];
        halfWeights.Store(first * 2u, f32tof16(updated[0]) | (f32tof16(updated[1]) << 16));
    }
}
