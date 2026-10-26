// The real-time clouds' layer at the render's size (docs/architecture.md §8.19): the pixels this frame
// marched (CloudMarch.hlsl, one of every 4 x 4) blend into their history, the others follow theirs back
// to the previous frame along the depth the clouds' light comes from, clamped to the march's
// neighbourhood so that nothing trails where the clouds changed. A pixel whose history lies off screen
// (what a moving camera brings in at the edges) is marched here at once: upsampled from the sparse march
// it showed as streaks along the edge for the 16 frames it took to fill in. Without any history (the
// first frame after a reset) the march is upsampled. The history is read through a Catmull-Rom filter.

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"
#include "CloudMarchShared.h"

[[vk::push_constant]] ConstantBuffer<CloudMarchConstants> g : register(b0);

#include "CloudMarch.hlsli"

static const float kCloudFreshBlend = 0.5f; // weight of a newly marched pixel against its history

// The history at `position` (pixels) through a Catmull-Rom filter, in 9 bilinear taps (the outer
// weights folded into the inner taps' positions): sampling it bilinearly at the fractional positions
// a drifting cloud or a moving camera reprojects to blurred it a little more every frame.
float4 SampleHistory(Texture2D<float4> history, float2 position, float2 size)
{
    float2 center = floor(position - 0.5f) + 0.5f;
    float2 f = position - center;
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);
    float2 w12 = w1 + w2;
    float2 p0 = (center - 1.0f) / size;
    float2 p3 = (center + 2.0f) / size;
    float2 p12 = (center + w2 / w12) / size;
    float4 sum = 0.0f;
    sum += history.SampleLevel(gLinearClamp, float2(p0.x, p0.y), 0.0f) * (w0.x * w0.y);
    sum += history.SampleLevel(gLinearClamp, float2(p12.x, p0.y), 0.0f) * (w12.x * w0.y);
    sum += history.SampleLevel(gLinearClamp, float2(p3.x, p0.y), 0.0f) * (w3.x * w0.y);
    sum += history.SampleLevel(gLinearClamp, float2(p0.x, p12.y), 0.0f) * (w0.x * w12.y);
    sum += history.SampleLevel(gLinearClamp, float2(p12.x, p12.y), 0.0f) * (w12.x * w12.y);
    sum += history.SampleLevel(gLinearClamp, float2(p3.x, p12.y), 0.0f) * (w3.x * w12.y);
    sum += history.SampleLevel(gLinearClamp, float2(p0.x, p3.y), 0.0f) * (w0.x * w3.y);
    sum += history.SampleLevel(gLinearClamp, float2(p12.x, p3.y), 0.0f) * (w12.x * w3.y);
    sum += history.SampleLevel(gLinearClamp, float2(p3.x, p3.y), 0.0f) * (w3.x * w3.y);
    return sum;
}

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
        // Where the clouds there were last frame, before the wind carried them on.
        float3 before = Xyz(f.camera.position) + d * depth - float3(g.windStepX, 0.0f, g.windStepZ);
        float2 previous = CameraProject(f.previousCamera, before, 1.0f, size);
        if (!all(previous >= 0.0f) || !all(previous < size))
        {
            if (!fresh)
            {
                float cycle = float(f.frameIndex / 16u);
                float ign = frac(52.9829189f * frac(0.06711056f * float(id.x) + 0.00583715f * float(id.y)));
                float2 u = frac(ign + cycle * float2(0.7548777f, 0.5698403f) + float2(0.0f, 0.5f));
                float3 ray = CameraRayDirection(f.camera, float2(id.xy) + u, size);
                CloudRay marchedHere = MarchClouds(f, Xyz(f.camera.position), ray, 1e30f, frac(ign + 0.25f),
                                                   frac(ign * 0.7548777f + cycle * 0.4142136f), g.steps, g.lightSamples);
                result = float4(marchedHere.light, marchedHere.transmittance);
                depth = marchedHere.depth;
            }
        }
        else
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
            float4 past = clamp(SampleHistory(history, previous, size), lo, hi);
            result = fresh ? lerp(past, current, kCloudFreshBlend) : past;
        }
    }
    RWTexture2D<float4> layer = ResourceDescriptorHeap[g.target];
    RWTexture2D<float> layerDepth = ResourceDescriptorHeap[g.targetDepth];
    layer[id.xy] = result;
    layerDepth[id.xy] = depth;
}
