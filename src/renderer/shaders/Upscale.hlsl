// Bilinear upscale of the scene color to the output, used by FrameRunner when the render resolution
// is below the output resolution and no Upscale feature (dlss) reconstructed the output this frame.

#include "air/renderer/shader/UpscaleShared.h"

[[vk::push_constant]] ConstantBuffer<UpscaleConstants> g : register(b0);

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.destinationWidth || id.y >= g.destinationHeight)
    {
        return;
    }
    Texture2D<float4> source = ResourceDescriptorHeap[g.source];
    RWTexture2D<float4> destination = ResourceDescriptorHeap[g.destination];

    // Pixel centres map onto pixel centres; taps are clamped to the source edges.
    float2 scale = float2(g.sourceWidth, g.sourceHeight) / float2(g.destinationWidth, g.destinationHeight);
    float2 p = (float2(id.xy) + 0.5f) * scale - 0.5f;
    float2 base = floor(p);
    float2 f = p - base;
    int2 maxCoord = int2(g.sourceWidth, g.sourceHeight) - 1;
    int2 c0 = clamp(int2(base), int2(0, 0), maxCoord);
    int2 c1 = clamp(int2(base) + 1, int2(0, 0), maxCoord);
    float4 top = lerp(source[int2(c0.x, c0.y)], source[int2(c1.x, c0.y)], f.x);
    float4 bottom = lerp(source[int2(c0.x, c1.y)], source[int2(c1.x, c1.y)], f.x);
    destination[id.xy] = lerp(top, bottom, f.y);
}
