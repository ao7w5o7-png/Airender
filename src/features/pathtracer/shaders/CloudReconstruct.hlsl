// The real-time clouds' layer at the render's size (docs/architecture.md §8.19): the pixels this frame
// marched (CloudMarch.hlsl, one of every 4 x 4) blend into their history, the others follow theirs back
// to the previous frame along the depth the clouds' light comes from, clamped to the march's
// neighbourhood so that nothing trails where the clouds changed. Without history (the first frame, off
// screen) the march is upsampled.

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"
#include "CloudMarchShared.h"

[[vk::push_constant]] ConstantBuffer<CloudMarchConstants> g : register(b0);

#include "CloudMarch.hlsli"

static const float kCloudFreshBlend = 0.4f; // weight of a newly marched pixel against its history

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.layerWidth || id.y >= g.layerHeight)
    {
        return;
    }
    FrameData f = LoadFrame();
    Texture2D<float4> marched = ResourceDescriptorHeap[g.marched];
    Texture2D<float> marchedDepth = ResourceDescriptorHeap[g.marchedDepth];
    uint2 offset = uint2(g.offset & 3u, g.offset >> 2);
    int2 cell = int2(id.xy) / 4;
    int2 last = int2(g.width, g.height) - 1;
    bool fresh = all((id.xy & 3u) == offset);
    // This frame's march around the pixel: upsampled (marched texel i is layer pixel 4 i + offset) and
    // its neighbourhood's range.
    float2 uv = ((float2(id.xy) - float2(offset)) * 0.25f + 0.5f) / float2(g.width, g.height);
    float4 current = fresh ? marched[uint2(min(cell, last))] : marched.SampleLevel(gLinearClamp, uv, 0.0f);
    float depth = fresh ? marchedDepth[uint2(min(cell, last))] : marchedDepth.SampleLevel(gLinearClamp, uv, 0.0f);
    float4 result = current;
    if (g.history != kInvalidIndex)
    {
        float2 size = float2(f.width, f.height);
        float3 d = CameraRayDirection(f.camera, float2(id.xy) + 0.5f, size);
        float2 previous = CameraProject(f.previousCamera, Xyz(f.camera.position) + d * depth, 1.0f, size);
        if (all(previous >= 0.0f) && all(previous < size))
        {
            float4 lo = current;
            float4 hi = current;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    float4 n = marched[uint2(clamp(cell + int2(dx, dy), int2(0, 0), last))];
                    lo = min(lo, n);
                    hi = max(hi, n);
                }
            }
            Texture2D<float4> history = ResourceDescriptorHeap[g.history];
            float4 past = clamp(history.SampleLevel(gLinearClamp, previous / size, 0.0f), lo, hi);
            result = fresh ? lerp(past, current, kCloudFreshBlend) : past;
        }
    }
    RWTexture2D<float4> layer = ResourceDescriptorHeap[g.target];
    RWTexture2D<float> layerDepth = ResourceDescriptorHeap[g.targetDepth];
    layer[id.xy] = result;
    layerDepth[id.xy] = depth;
}
