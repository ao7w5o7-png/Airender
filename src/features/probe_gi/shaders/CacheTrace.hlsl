// Probe GI radiance cache (ProbeGiShared.h): the rays of the probes this frame updates, one group per probe,
// one thread per octahedral texel, its direction jittered within the texel every frame. A ray traced through
// the voxels from the probe takes the light the voxels it hits reflect towards it, or the sky's when it
// leaves the cascades. Writes radiance and distance (negative: it hit the back of a surface) per ray.
// Every level in one dispatch (group y: the level; its CascadeData::cacheProbes probes from where the last
// frame's left off, or all).
//   GiConstants: probeCount = the largest number of probes of a level this frame (the rays' stride), rays.

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

[numthreads(kCacheTexels, kCacheTexels, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint level = group.y;
    if (level >= f.cascadeCount || f.cascades[level].valid == 0u)
    {
        return;
    }
    CascadeData c = f.cascades[level];
    uint count = c.cacheProbes == 0u ? kCacheProbes : c.cacheProbes;
    if (group.x >= count)
    {
        return;
    }
    uint first = c.cacheProbes == 0u ? 0u : (f.frameIndex * c.cacheProbes) % kCacheProbes;
    uint index = (first + group.x) % kCacheProbes;
    // The probe's world coordinates from its storage coordinates.
    int3 origin = CacheOrigin(c);
    uint3 storage = uint3(index % kCacheResolution, (index / kCacheResolution) % kCacheResolution,
                          index / (kCacheResolution * kCacheResolution));
    int3 local = int3((storage - uint3(origin & int(kCacheResolution - 1u))) & (kCacheResolution - 1u));
    float3 position = CacheProbePosition(c, origin + local);
    uint seed = Pcg(index ^ Pcg(f.frameIndex * 8u + level));
    float2 jitter = Rand2(seed);
    float3 direction = CacheDirection((float2(thread.xy) + jitter) / float(kCacheTexels));
    VoxelHit hit = TraceVoxels(f, position, direction, 0.0f, 1e30f);
    float3 radiance;
    float distance;
    if (hit.hit)
    {
        bool found;
        bool backface;
        radiance = VoxelRadiance(f, hit.cascade, hit.voxel, direction, found, backface);
        distance = backface ? -hit.t : hit.t;
    }
    else
    {
        radiance = SkyLight(f, direction);
        distance = CacheSpacing(c) * float(kCacheResolution); // past the level: as good as infinite
    }
    RWStructuredBuffer<float4> rays = ResourceDescriptorHeap[g.rays];
    rays[(level * g.probeCount + group.x) * kCacheTexels * kCacheTexels + thread.y * kCacheTexels + thread.x] =
        float4(radiance, distance);
}
