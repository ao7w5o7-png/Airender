// Probe GI voxelization (Voxelize.hlsli): per surface voxel voxelized in this update, its faces' averages
// (albedo, normal, how much of the voxel they cover) packed into its SurfaceVoxel, and a first guess of their
// light until VoxelLight.hlsl lights them (age 0): where the cascade moved, what the next coarser cascade's
// voxels reflect there; else 0 (VoxelLight.hlsl lights a cascade fully in the frame it was built). One thread
// per voxel.

#include "Voxelize.hlsli"
#include "VoxelTrace.hlsli"

static const float kMinCoverage = 0.02f;

[numthreads(4, 4, 4)]
void Main(uint3 id : SV_DispatchThreadID)
{
    RWByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    if (all(id == 0u))
    {
        // VoxelizeAllocate may have taken the free ring's head past its tail.
        counters.Store(kCounterFreeHead, min(counters.Load(kCounterFreeHead), counters.Load(kCounterFreeTail)));
    }
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    CascadeData c = f.cascades[g.cascade];
    int3 local = LocalOf(id, CascadeOrigin(c));
    if (Kept(local))
    {
        return;
    }
    Texture3D<uint> voxelIndex = ResourceDescriptorHeap[g.voxelIndex];
    uint slot = voxelIndex[id];
    if (slot == 0u)
    {
        return;
    }
    --slot;
    RWStructuredBuffer<uint> acc = ResourceDescriptorHeap[g.accumulators];
    RWStructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[g.voxels];
    float3 centre = (float3(CascadeOrigin(c) + local) + 0.5f) * c.voxelSize;
    uint coarse = g.cascade + 1u;
    bool guess = g.incremental != 0u && coarse < f.cascadeCount && f.cascades[coarse].valid != 0u &&
                 InCascade(f.cascades[coarse], centre, kCascadeMargin + 1.0f);
    SurfaceVoxel v = voxels[slot];
    uint mask = 0u;
    for (uint face = 0u; face < 6u; ++face)
    {
        uint base = slot * kAccumulatorStride + face * kAccumulatorFace;
        float weight = float(acc[base + 3u]);
        float coverage = weight / kAccumulatorScale;
        v.radiance[face] = 0u;
        v.direct[face] = 0u;
        if (coverage < kMinCoverage)
        {
            v.albedo[face] = 0u;
            v.normal[face] = 0u;
            continue;
        }
        float3 albedo = float3(acc[base], acc[base + 1u], acc[base + 2u]) / weight;
        float3 n = float3(asint(acc[base + 4u]), asint(acc[base + 5u]), asint(acc[base + 6u]));
        n = dot(n, n) > 0.0f ? normalize(n) : float3(0.0f, 1.0f, 0.0f);
        float offset = clamp(float(asint(acc[base + 7u])) / weight, -1.0f, 1.0f);
        v.albedo[face] = PackUnorm4(float4(albedo, saturate(coverage)));
        v.normal[face] = PackNormalOffset(n, offset);
        if (guess)
        {
            // The light the coarser voxels around the surface reflect the face's way (rays coming against it).
            float3 surface = centre + n * (offset * c.voxelSize);
            bool found;
            bool backface;
            float3 light = VoxelRadiance(f, coarse, surface / f.cascades[coarse].voxelSize, -kFaceAxes[face], found, backface);
            v.radiance[face] = PackRadiance(light);
        }
        mask |= 1u << face;
    }
    v.coord = (v.coord & 0x00FFFFFFu) | (mask << 24);
    v.age = 0u;
    voxels[slot] = v;
    if (g.incremental != 0u && mask != 0u)
    {
        // New where the cascade moved: lit first (kFreshFrames).
        uint count = WaveActiveCountBits(true);
        uint first = 0u;
        if (WaveIsFirstLane())
        {
            counters.InterlockedAdd(kCounterFreshHead, count, first);
        }
        uint head = WaveReadLaneFirst(first) + WavePrefixCountBits(true);
        RWStructuredBuffer<uint2> fresh = ResourceDescriptorHeap[g.fresh];
        fresh[head % kFreshCapacity] = uint2(slot, f.frameIndex);
    }
}
