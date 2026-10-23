// Accumulation buffer -> display: average, exposure, ACES fit tonemap.
//   dxc -T cs_6_6 -E Main Resolve.hlsl

#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<ResolveConstants> g : register(b0);

float3 AcesFitted(float3 x) {
    // Narkowicz 2015
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g.width || id.y >= g.height) {
        return;
    }
    Texture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulationSrv];
    RWTexture2D<float4> output = ResourceDescriptorHeap[g.outputUav];

    float4 sum = accumulation[id.xy];
    float3 color = sum.a > 0.0f ? sum.rgb / sum.a : 0.0f;
    output[id.xy] = float4(AcesFitted(color * g.exposure), 1.0f);
}
