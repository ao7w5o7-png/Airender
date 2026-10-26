// The real-time clouds over the camera's rays (docs/architecture.md §8.19): one pixel of every 4 x 4
// block of the layer (the render's size), a different one each frame, along a ray jittered within it.
// CloudReconstruct.hlsl fills in the rest.

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"
#include "CloudMarchShared.h"

[[vk::push_constant]] ConstantBuffer<CloudMarchConstants> g : register(b0);

#include "CloudMarch.hlsli"

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    uint2 layerPixel = id.xy * 4u + uint2(g.offset & 3u, g.offset >> 2);
    FrameData f = LoadFrame();
    uint seed = Pcg(id.x + Pcg(id.y + Pcg(f.frameIndex ^ 0x434C4F44u)));
    float2 u = Rand2(seed);
    float2 at = float2(min(layerPixel, uint2(g.layerWidth, g.layerHeight) - 1u)) + u;
    float3 d = CameraRayDirection(f.camera, at, float2(f.width, f.height));
    float offset = Rand(seed);
    CloudRay ray = MarchClouds(f, Xyz(f.camera.position), d, 1e30f, offset, Rand(seed), g.steps, g.lightSamples);
    RWTexture2D<float4> marched = ResourceDescriptorHeap[g.target];
    RWTexture2D<float> depth = ResourceDescriptorHeap[g.targetDepth];
    marched[id.xy] = float4(ray.light, ray.transmittance);
    depth[id.xy] = ray.depth;
}
