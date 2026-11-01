// Probe GI radiance cache (ProbeGiShared.h): after a cascade moved, its level's probes that now stand where
// the level did not reach before (toroidal storage: their slots held probes elsewhere) start over: invalid,
// so that their first trace replaces them. The others keep their light. One thread per probe.
//   GiConstants: cascade = the level, input0..2 = the level's previous origin in probe coordinates (int),
//   input3 = 1 when there was none (every probe starts over).

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= kCacheProbes)
    {
        return;
    }
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    CascadeData c = f.cascades[g.cascade];
    uint index = id.x;
    int3 origin = CacheOrigin(c);
    uint3 storage = uint3(index % kCacheResolution, (index / kCacheResolution) % kCacheResolution,
                          index / (kCacheResolution * kCacheResolution));
    int3 local = int3((storage - uint3(origin & int(kCacheResolution - 1u))) & (kCacheResolution - 1u));
    int3 probe = origin + local;
    int3 previous = int3(asint(g.input0), asint(g.input1), asint(g.input2));
    int3 inPrevious = probe - previous;
    bool kept = g.input3 == 0u && all(inPrevious >= 0) && all(inPrevious < int(kCacheResolution));
    if (kept)
    {
        return;
    }
    RWTexture2D<float4> atlas = ResourceDescriptorHeap[g.input4];
    RWTexture2D<float2> moments = ResourceDescriptorHeap[g.input5];
    for (uint y = 0u; y < kCacheTexels; ++y)
    {
        for (uint x = 0u; x < kCacheTexels; ++x)
        {
            uint2 texel = CacheAtlasTexel(index, uint2(x, y));
            atlas[texel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            moments[texel] = float2(0.0f, 0.0f);
        }
    }
    RWStructuredBuffer<ProbeSh> probes = ResourceDescriptorHeap[g.input6];
    probes[index] = (ProbeSh)0;
}
