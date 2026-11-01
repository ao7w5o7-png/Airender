// Probe GI screen probes (ProbeGiShared.h): one probe per tile of probeTile^2 pixels, on the surface a pixel
// of the tile sees. The pixel moves every frame (R2), so that over frames the probes cover their tiles; a
// tile whose chosen pixel shows the sky tries its centre and corners. One thread per tile.
//   GiConstants: input0 view depth, input1 normal + roughness (Texture2D<float4>), output probes
//   (RWStructuredBuffer<ScreenProbe>; the adaptive probes' rows cleared), counters the adaptive probes' count
//   (RWByteAddressBuffer at 0: cleared).

#include "GiCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kSkyDepth = 65000.0f;

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.probesX || id.y >= f.probesY + f.adaptiveRows)
    {
        return;
    }
    RWStructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.output];
    if (id.y >= f.probesY)
    {
        // The adaptive probes' rows: none until ProbeAdapt.hlsl places them.
        probes[id.y * f.probesX + id.x] = (ScreenProbe)0;
        if (all(id.xy == uint2(0u, f.probesY)))
        {
            RWByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
            counters.Store(0, 0u);
        }
        return;
    }
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input1];
    uint tile = f.probeTile;
    uint2 corner = id.xy * tile;
    uint2 last = uint2(f.width - 1u, f.height - 1u);
    float2 offset = R2(f.frameIndex + (id.x * 7u + id.y * 13u) % 3u);
    uint2 candidates[5] = {uint2(offset * float(tile)), uint2(tile / 2u, tile / 2u), uint2(0u, 0u),
                           uint2(tile - 1u, tile - 1u), uint2(tile - 1u, 0u)};
    ScreenProbe probe = (ScreenProbe)0;
    for (uint i = 0u; i < 5u; ++i)
    {
        uint2 pixel = min(corner + candidates[i], last);
        float z = depth.Load(int3(pixel, 0));
        if (z < kSkyDepth && z > 0.0f)
        {
            float3 direction;
            probe.position = PixelPosition(f, pixel, z, direction);
            float3 n = normalRoughness.Load(int3(pixel, 0)).xyz;
            n = dot(n, n) > 0.0f ? normalize(n) : -direction;
            probe.normal = PackNormal(n);
            probe.depth = z;
            probe.pixel = pixel.x | (pixel.y << 16);
            probe.footprint = PixelFootprint(f, z);
            break;
        }
    }
    probes[id.y * f.probesX + id.x] = probe;
}
