// Probe GI voxelization (Voxelize.hlsli): a slot of the surface voxel pool for every marked voxel (a free
// one while the ring of free slots holds any, else a new one), its accumulators cleared. One thread per voxel.

#include "Voxelize.hlsli"

[numthreads(4, 4, 4)]
void Main(uint3 id : SV_DispatchThreadID)
{
    RWTexture3D<uint> voxelIndex = ResourceDescriptorHeap[g.voxelIndex];
    if (voxelIndex[id] != kVoxelMarked)
    {
        return;
    }
    RWByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    RWStructuredBuffer<uint> freeSlots = ResourceDescriptorHeap[g.freeSlots];
    // One atomic per wave for the free slots (the head may pass the tail: VoxelizeResolve puts it back), one for
    // new ones.
    uint marked = WaveActiveCountBits(true);
    uint first = 0u;
    if (WaveIsFirstLane())
    {
        counters.InterlockedAdd(kCounterFreeHead, marked, first);
    }
    uint head = WaveReadLaneFirst(first) + WavePrefixCountBits(true);
    bool reuse = head < counters.Load(kCounterFreeTail);
    uint fresh = WaveActiveCountBits(!reuse);
    uint freshIndex = WavePrefixCountBits(!reuse);
    uint firstFresh = 0u;
    if (WaveIsFirstLane() && fresh > 0u)
    {
        counters.InterlockedAdd(kCounterVoxels, fresh, firstFresh);
    }
    firstFresh = WaveReadLaneFirst(firstFresh);
    uint slot = reuse ? freeSlots[head % g.voxelCapacity] : firstFresh + freshIndex;
    if (slot >= g.voxelCapacity)
    {
        voxelIndex[id] = 0u;
        counters.InterlockedAdd(kCounterVoxelOverflow, 1u);
        return;
    }
    voxelIndex[id] = slot + 1u;
    RWStructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[g.voxels];
    SurfaceVoxel v = (SurfaceVoxel)0;
    v.coord = id.x | (id.y << 8) | (id.z << 16);
    voxels[slot] = v;
    RWStructuredBuffer<uint> accumulators = ResourceDescriptorHeap[g.accumulators];
    for (uint i = 0u; i < kAccumulatorStride; ++i)
    {
        accumulators[slot * kAccumulatorStride + i] = 0u;
    }
}
