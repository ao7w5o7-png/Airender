// The real-time clouds' shadow map (CloudMaps.hlsli, docs/architecture.md §8.19): over one tile of the
// clouds, the optical depth along the sun from each point of their base up through the layer. Rendered
// when the clouds or the sun change, and as the wind moves their detail over the base density.

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
    FrameData f = LoadFrame();
    CloudData c = f.clouds;
    float3 sun = f.sky.sunDirection;
    float depth = 0.0f;
    if (c.enabled != 0u && sun.y > 0.0f)
    {
        // A sun grazing the layer would cross it for hundreds of kilometres: its rays leave at 3 degrees.
        sun = normalize(float3(sun.x, max(sun.y, 0.05f), sun.z));
        // A point of the base, at its altitude. (The tile is small next to the planet: its texels serve
        // the base wherever it repeats.)
        float x = (float(id.x) + 0.5f) / float(g.width) * c.tileSize + c.baseOffsetX;
        float z = (float(id.y) + 0.5f) / float(g.height) * c.tileSize + c.baseOffsetZ;
        float3 o = float3(x, c.bottom - (x * x + z * z) * (0.5f / kCloudPlanetRadius), z);
        CloudMarch m;
        if (BeginCloudMarch(c, o, sun, 1e30f, m))
        {
            float t0;
            float t1;
            float majorant;
            while (NextCloudSegment(c, m, t0, t1, majorant))
            {
                if (majorant <= 0.0f)
                {
                    continue;
                }
                // Midpoints of steps of at most 25 m through the cell.
                uint steps = clamp(uint(ceil((t1 - t0) / 25.0f)), 1u, 64u);
                float dt = (t1 - t0) / float(steps);
                for (uint i = 0u; i < steps; ++i)
                {
                    depth += CloudDensity(c, o + sun * (t0 + (float(i) + 0.5f) * dt)) * dt;
                }
            }
        }
        depth *= c.extinction;
    }
    RWTexture2D<float> map = ResourceDescriptorHeap[g.target];
    map[id.xy] = depth;
}
