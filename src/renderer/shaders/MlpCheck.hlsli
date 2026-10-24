// The network on the GPU for renderer::CheckNeuralNetwork (MlpCheckConstants): a thread per sample
// evaluates it, backpropagates the sample's loss gradient (scaled as training kernels scale it) into
// the network's gradients and writes the outputs and the input gradient. Compiled twice: MlpCheck.hlsl
// on the matrix units (AIR_MLP_LINALG 1) and MlpCheckPortable.hlsl in float32.

#include "air/renderer/shader/MlpShared.h"

[[vk::push_constant]] ConstantBuffer<MlpCheckConstants> g : register(b0);

float4 MlpLoadWeights4(uint index)
{
    ByteAddressBuffer parameters = ResourceDescriptorHeap[g.network.parameters];
    return asfloat(parameters.Load4(index * 4u));
}

float2 MlpLoadFeatures(uint index)
{
    ByteAddressBuffer parameters = ResourceDescriptorHeap[g.network.parameters];
    return asfloat(parameters.Load2(index * 4u));
}

#include "air/renderer/shader/MlpGradients.hlsli"
#if AIR_MLP_LINALG
#include "air/renderer/shader/MlpLinAlg.hlsli"
#endif

// Every thread runs the network (threads past the samples on the last one, with no gradient): the
// matrix units see whole waves and the float32 path's wave sums need every lane.
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    bool valid = id.x < g.samples;
    uint sample = min(id.x, g.samples - 1u);
    ByteAddressBuffer inputs = ResourceDescriptorHeap[g.inputs];
    ByteAddressBuffer outputGradients = ResourceDescriptorHeap[g.outputGradients];
    MlpVector x;
    MlpVector dy;
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        x.v[i] = asfloat(inputs.Load4((sample * kMlpWidth + i * 4u) * 4u));
        dy.v[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (uint o = 0u; o < kMlpOutputs / 4u; ++o)
    {
        float4 d = asfloat(outputGradients.Load4((sample * kMlpOutputs + o * 4u) * 4u));
        dy.v[o] = valid ? d * kMlpLossScale : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
#if AIR_MLP_LINALG
    MlpLinAlgForward f = MlpLinAlgEvaluate(g.network, MlpToHalf(x));
    MlpHalfOutput dyHalf;
    for (uint k = 0u; k < kMlpOutputs; ++k)
    {
        dyHalf[k] = (half)MlpGet(dy, k);
    }
    MlpVector dx = MlpFromHalf(MlpLinAlgBackpropagate(g.network, f, dyHalf));
    float outputs[kMlpOutputs];
    for (uint k = 0u; k < kMlpOutputs; ++k)
    {
        outputs[k] = f.output[k];
    }
#else
    MlpForward f = MlpEvaluate(x);
    MlpBackward b = MlpDifferentiate(f, dy);
    MlpAccumulateWeightGradients(g.network, f, b);
    MlpVector dx = b.inputGradient;
    float outputs[kMlpOutputs];
    for (uint k = 0u; k < kMlpOutputs; ++k)
    {
        outputs[k] = MlpGet(f.output, k);
    }
#endif
    if (valid)
    {
        RWByteAddressBuffer outputBuffer = ResourceDescriptorHeap[g.outputs];
        RWByteAddressBuffer inputGradients = ResourceDescriptorHeap[g.inputGradients];
        for (uint k = 0u; k < kMlpOutputs; ++k)
        {
            outputBuffer.Store((sample * kMlpOutputs + k) * 4u, asuint(outputs[k]));
        }
        for (uint i = 0u; i < kMlpWidth; ++i)
        {
            inputGradients.Store((sample * kMlpWidth + i) * 4u, asuint(MlpGet(dx, i) / kMlpLossScale));
        }
    }
}
