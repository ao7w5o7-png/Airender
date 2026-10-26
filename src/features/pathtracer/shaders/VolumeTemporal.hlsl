// The volumes' layer filtered over frames (docs/architecture.md §8.13): what the march lights changes
// from frame to frame (its jitter, the maps it reads in turn, the blocks that share their light), which
// accumulation or a denoiser of the whole frame averages. Without them (in real time under a denoiser
// of the surfaces alone, or none) this pass does: it follows the layer's point back to the previous
// frame along the depth its light comes from, clamps the history found there to the 3 x 3 pixels around
// (so that it cannot drag what moved), blends in this frame and lays the result over the surfaces.

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<VolumeTemporalConstants> g : register(b0);

SamplerState gLinearClamp : register(s0);

#include "Camera.hlsli"

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    if (id.x >= f.width || id.y >= f.height)
    {
        return;
    }
    uint2 pixel = id.xy;
    float2 size = float2(f.width, f.height);
    Texture2D<float4> layer = ResourceDescriptorHeap[g.layer];
    Texture2D<float> layerDepth = ResourceDescriptorHeap[g.layerDepth];
    float4 current = layer[pixel];
    float depth = layerDepth[pixel];
    float4 result = current;
    if (depth > 0.0f && g.history != kInvalidIndex)
    {
        float3 d = CameraRayDirection(f.camera, float2(pixel) + f.jitter + 0.5f, size);
        float3 x = Xyz(f.camera.position) + d * depth;
        float2 previous = CameraProject(f.previousCamera, x, 1.0f, size);
        if (all(previous >= 0.0f) && all(previous < size))
        {
            float4 lo = current;
            float4 hi = current;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int2 q = clamp(int2(pixel) + int2(dx, dy), int2(0, 0), int2(size) - 1);
                    float4 n = layer[uint2(q)];
                    lo = min(lo, n);
                    hi = max(hi, n);
                }
            }
            Texture2D<float4> history = ResourceDescriptorHeap[g.history];
            float4 past = history.SampleLevel(gLinearClamp, previous / size, 0.0f);
            result = lerp(clamp(past, lo, hi), current, g.blend);
        }
    }
    RWTexture2D<float4> filtered = ResourceDescriptorHeap[g.filtered];
    filtered[pixel] = result;
    if (g.surfaceTransmittance != kInvalidIndex)
    {
        RWTexture2D<float> surfaceTransmittance = ResourceDescriptorHeap[g.surfaceTransmittance];
        surfaceTransmittance[pixel] = result.a;
    }
    Texture2D<float4> surface = ResourceDescriptorHeap[g.surface];
    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
    float samples = float(g.samplesPerPixel);
    float4 value = float4(result.rgb * samples + surface[pixel].rgb * result.a, samples);
    accumulation[pixel] = (g.reset != 0u ? float4(0.0f, 0.0f, 0.0f, 0.0f) : accumulation[pixel]) + value;
}
