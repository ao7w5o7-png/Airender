// Example feature: blend the pattern over the frame output.   dxc -T cs_6_6 -E Main Composite.hlsl -Fo Composite.cs.dxil

#include "ExampleShared.h"

[[vk::push_constant]] ConstantBuffer<CompositeConstants> g : register(b0);

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    Texture2D<float4> pattern = ResourceDescriptorHeap[g.patternSrv];
    RWTexture2D<float4> output = ResourceDescriptorHeap[g.outputUav];
    output[id.xy] = lerp(output[id.xy], pattern[id.xy], g.opacity);
}
