// Probe GI voxelization (Voxelize.hlsli): where the cascade is voxelized again (all of it, or what it did not
// cover before it moved), the distances to infinity and the voxel index to empty, the slots of the voxels
// there before (left behind by the move) into the ring of free slots; the bricks' occupancy and the update's
// counters to 0 (all of them when everything is voxelized again). One thread per voxel.

#include "Voxelize.hlsli"

[numthreads(4, 4, 4)]
void Main(uint3 id : SV_DispatchThreadID)
{
    CascadeData c = LoadCascade();
    RWByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    if (all(id < kBrickResolution))
    {
        RWTexture3D<uint> occupancy = ResourceDescriptorHeap[g.occupancy];
        occupancy[id] = 0u;
    }
    if (all(id == 0u))
    {
        counters.Store(kCounterWorkItems, 0u);
        counters.Store(kCounterWorkOverflow, 0u);
        counters.Store(kCounterVoxelOverflow, 0u);
        if (g.incremental == 0u)
        {
            counters.Store(kCounterVoxels, 0u);
            counters.Store(kCounterFreeHead, 0u);
            counters.Store(kCounterFreeTail, 0u);
        }
    }
    if (Kept(LocalOf(id, Origin(c))))
    {
        return;
    }
    RWTexture3D<uint> distances = ResourceDescriptorHeap[g.distances];
    RWTexture3D<uint> voxelIndex = ResourceDescriptorHeap[g.voxelIndex];
    distances[id] = 0x7F7FFFFFu; // the largest float
    uint old = voxelIndex[id];
    voxelIndex[id] = 0u;
    if (g.incremental != 0u && old != 0u)
    {
        // The voxel that was here lies behind the cascade now: its slot is free, the voxel without faces (unlit).
        RWStructuredBuffer<uint> freeSlots = ResourceDescriptorHeap[g.freeSlots];
        RWStructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[g.voxels];
        uint freed = WaveActiveCountBits(true);
        uint first = 0u;
        if (WaveIsFirstLane())
        {
            counters.InterlockedAdd(kCounterFreeTail, freed, first);
        }
        uint tail = WaveReadLaneFirst(first) + WavePrefixCountBits(true);
        freeSlots[tail % g.voxelCapacity] = old - 1u;
        voxels[old - 1u].coord = 0u;
    }
}
