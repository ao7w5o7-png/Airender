// Probe GI screen probes (ProbeGiShared.h): adaptive probes where the tiles' probes fit nothing. Per tile, at a
// pixel jittered within each of its 2 x 2 quarters (R2), the 4 tiles' probes around are weighed as the gather
// weighs them (ProbeFit.hlsli); where they fit less than kAdaptiveFit and no probe this tile placed already fits, a
// probe of its own (a slot in the rows after the tiles', GiFrame::adaptiveRows, while any are left). Edges, small
// objects and foliage otherwise take the search's nearest fitting probe or the radiance cache's coarse light
// (sponza: 6% and 3% of the pixels). The tile's header lists its probes for the gather. One thread per tile.
//   GiConstants: input0 view depth, input1 normal + roughness, output probes (RWStructuredBuffer<ScreenProbe>),
//   input5 the tiles' headers (RWStructuredBuffer<uint4>: probe index + 1, 0 none), counters the adaptive probes'
//   count (RWByteAddressBuffer at 0).

#include "GiCommon.hlsli"
#include "ProbeFit.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kSkyDepth = 65000.0f;
static const float kAdaptiveFit = 0.05f;

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.probesX || id.y >= f.probesY)
    {
        return;
    }
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input1];
    RWStructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.output];
    RWStructuredBuffer<uint4> headers = ResourceDescriptorHeap[g.input5];
    RWByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    uint tileIndex = id.y * f.probesX + id.x;
    uint half = max(f.probeTile / 2u, 1u);
    uint2 last = uint2(f.width - 1u, f.height - 1u);
    uint capacity = f.probesX * f.adaptiveRows;
    uint placed[4] = {0u, 0u, 0u, 0u};
    uint count = 0u;
    for (uint q = 0u; q < 4u; ++q)
    {
        float2 jitter = R2(f.frameIndex * 4u + q + (id.x * 7u + id.y * 13u) % 5u);
        uint2 pixel = min(id.xy * f.probeTile + uint2(q & 1u, q >> 1) * half + uint2(jitter * float(half)), last);
        float z = depth.Load(int3(pixel, 0));
        if (!(z < kSkyDepth) || z <= 0.0f)
        {
            continue;
        }
        float3 view;
        float3 p = PixelPosition(f, pixel, z, view);
        float3 n = normalRoughness.Load(int3(pixel, 0)).xyz;
        n = dot(n, n) > 0.0f ? normalize(n) : -view;
        float footprint = PixelFootprint(f, z);
        float3 plane = DepthNormal(f, depth, pixel, p, z, view, n);
        // The tiles' probes around, as the gather weighs them.
        float2 tile = (float2(pixel) + 0.5f) / float(f.probeTile) - 0.5f;
        int2 base = int2(floor(tile));
        float2 w1 = tile - float2(base);
        float fit = 0.0f;
        for (uint k = 0u; k < 4u; ++k)
        {
            int2 corner = int2(k & 1u, k >> 1);
            int2 t = clamp(base + corner, int2(0, 0), int2(f.probesX - 1u, f.probesY - 1u));
            float2 wc = select(corner != 0, w1, 1.0f - w1);
            fit += wc.x * wc.y * FitWeight(p, n, plane, footprint, probes[uint(t.y) * f.probesX + uint(t.x)]);
        }
        for (uint i = 0u; i < count; ++i)
        {
            fit += FitWeight(p, n, plane, footprint, probes[placed[i] - 1u]);
        }
        if (fit >= kAdaptiveFit)
        {
            continue;
        }
        uint slot;
        counters.InterlockedAdd(0, 1u, slot);
        if (slot >= capacity)
        {
            break;
        }
        uint index = f.probesX * f.probesY + slot;
        ScreenProbe probe;
        probe.position = p;
        probe.normal = PackNormal(n);
        probe.depth = z;
        probe.pixel = pixel.x | (pixel.y << 16);
        probe.footprint = footprint;
        probe.pad0 = 0u;
        probes[index] = probe;
        placed[count++] = index + 1u;
    }
    headers[tileIndex] = uint4(placed[0], placed[1], placed[2], placed[3]);
}
