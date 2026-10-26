// The real-time clouds' sky map (CloudMaps.hlsli, docs/architecture.md §8.19): what the clouds add to
// and leave of the sky towards each direction of the upper hemisphere, seen from the camera, for the
// rays that leave the scene after a bounce.

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"
#include "CloudMarchShared.h"

[[vk::push_constant]] ConstantBuffer<CloudMarchConstants> g : register(b0);

#include "CloudMarch.hlsli"
#define AIR_PT_CLOUD_MAPS_MAPPING_ONLY 1 // the mapping, not the lookups
#include "CloudMaps.hlsli"

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    FrameData f = LoadFrame();
    float3 d = CloudSkyMapDirection((float2(id.xy) + 0.5f) / float2(g.width, g.height));
    float4 value = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (d.y > 0.0f)
    {
        // The same steps every frame: the map stays still where its light does (bounce light is soft).
        CloudRay ray = MarchClouds(f, Xyz(f.camera.position), d, 1e30f, 0.5f, g.steps, g.lightSamples);
        value = float4(ray.light, ray.transmittance);
    }
    RWTexture2D<float4> map = ResourceDescriptorHeap[g.target];
    map[id.xy] = value;
}
