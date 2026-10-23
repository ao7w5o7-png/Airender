// Example feature: procedural pattern.   dxc -T cs_6_6 -E Main Generate.hlsl -Fo Generate.cs.dxil

#include "ExampleShared.h"

[[vk::push_constant]] ConstantBuffer<GenerateConstants> g : register(b0);

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    RWTexture2D<float4> output = ResourceDescriptorHeap[g.outputUav];
    float2 uv = (float2(id.xy) + 0.5f) / float2(g.width, g.height);
    float wave = 0.5f + 0.5f * sin((uv.x + uv.y) * g.scale + g.time);
    output[id.xy] = float4(uv, wave, 1.0f);
}
