// Probe GI voxelization (Voxelize.hlsli): per voxel voxelized in this update, the distance into the cascade's
// R8 field (distance / kVoxelBand, the band's edge where no triangle came closer); per surface voxel of the
// cascade, the brick it lies in is occupied.

#include "Voxelize.hlsli"

[numthreads(4, 4, 4)]
void Main(uint3 id : SV_DispatchThreadID)
{
    CascadeData c = LoadCascade();
    RWTexture3D<uint> voxelIndex = ResourceDescriptorHeap[g.voxelIndex];
    if (voxelIndex[id] != 0u)
    {
        RWTexture3D<uint> occupancy = ResourceDescriptorHeap[g.occupancy];
        occupancy[id / kVoxelBrick] = 1u;
    }
    if (Kept(LocalOf(id, Origin(c))))
    {
        return;
    }
    RWTexture3D<uint> distances = ResourceDescriptorHeap[g.distances];
    RWTexture3D<float> sdf = ResourceDescriptorHeap[g.sdf];
    sdf[id] = saturate(asfloat(distances[id]) / kVoxelBand);
}