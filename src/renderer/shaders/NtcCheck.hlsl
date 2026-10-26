// air-shader-target: cs_6_6
// renderer::CheckNeuralTextures (aircli ntc-check): a neural texture set decoded at every texel by
// LibNTC's shader code, as the path tracer decodes it on sample, in the colour spaces it is stored in.
// Built without LibNTC's shaders it writes zeros.

#include "air/renderer/shader/NeuralTextureShared.h"
#include "air/renderer/shader/SceneShared.h"

[[vk::push_constant]] ConstantBuffer<NeuralTextureCheckConstants> g : register(b0);

SamplerState gLinearWrap : register(s2);

#if __has_include("libntc/shaders/Inference.hlsli")
#define AIR_NTC 1
#include "libntc/shaders/Inference.hlsli"
#else
#define AIR_NTC 0
#endif

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    float values[16];
    [unroll]
    for (int i = 0; i < 16; ++i)
    {
        values[i] = 0.0f;
    }
#if AIR_NTC
    ByteAddressBuffer data = ResourceDescriptorHeap[g.data];
    Texture2DArray<float4> latents = ResourceDescriptorHeap[g.latents];
    NtcTextureSetConstants c = data.Load<NtcTextureSetConstants>(0);
    NtcSampleTextureSet(c, latents, gLinearWrap, data, kNeuralTextureConstantBytes, int2(id.xy), 0, false, values);
#endif
    RWByteAddressBuffer outputs = ResourceDescriptorHeap[g.outputs];
    uint texel = id.y * g.width + id.x;
    [unroll]
    for (uint channel = 0u; channel < kNeuralTextureChannels; ++channel)
    {
        outputs.Store((texel * kNeuralTextureChannels + channel) * 4u, asuint(values[channel]));
    }
}
