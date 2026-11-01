// Probe GI voxelization (Voxelize.hlsli): per work item, a group of 64 threads, one per voxel of its
// brick: the exact distance from the voxel's centre to the triangle (atomic minimum, the band), and the
// voxels it may cross are marked as surface voxels. Cutouts count only where their closest point is solid.

#include "Voxelize.hlsli"

[numthreads(64, 1, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    CascadeData c = LoadCascade();
    ByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    StructuredBuffer<uint2> items = ResourceDescriptorHeap[g.workItems];
    RWTexture3D<uint> distances = ResourceDescriptorHeap[g.distances];
    RWTexture3D<uint> voxelIndex = ResourceDescriptorHeap[g.voxelIndex];
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[g.materials];
    uint count = min(counters.Load(kCounterWorkItems), g.workCapacity);
    float inv = 1.0f / c.voxelSize;
    int3 origin = Origin(c);
    uint3 inBrick = uint3(thread.x & 3u, (thread.x >> 2) & 3u, thread.x >> 4);
    for (uint item = group.x; item < count; item += kGroupsMax)
    {
        uint2 work = items[item];
        Triangle tri = LoadTriangle(work.x);
        uint3 local = UnpackBrick(work.y) * kVoxelBrick + inBrick;
        int3 voxel = origin + int3(local);
        float3 p = (float3(voxel) + 0.5f) * c.voxelSize;
        float3 w = ClosestBarycentrics(p, tri.p0, tri.p1, tri.p2);
        float3 q = tri.p0 * w.x + tri.p1 * w.y + tri.p2 * w.z;
        float d = length(p - q) * inv;
        if (d >= kVoxelBand)
        {
            continue;
        }
        MaterialData m = materials[tri.material];
        if (m.alphaCutoff > 0.0f)
        {
            float2 uv = tri.uv0 * w.x + tri.uv1 * w.y + tri.uv2 * w.z;
            float footprint = 0.0f; // the finest level: a cutout's alpha is a hard edge
            if (VoxelMaterial(m, uv, footprint).w <= 0.0f)
            {
                continue;
            }
        }
        uint3 s = Storage(voxel);
        InterlockedMin(distances[s], asuint(d));
        if (d < kSurfaceDistance)
        {
            voxelIndex[s] = kVoxelMarked;
        }
    }
}
