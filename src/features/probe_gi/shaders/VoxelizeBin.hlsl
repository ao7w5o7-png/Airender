// Probe GI voxelization (Voxelize.hlsli): per triangle, a work item for every brick of the cascade that
// its band (kVoxelBand voxels around it) reaches, but those the update keeps; flagged kWorkNear where the
// triangle may cross the brick's voxels. Grid-stride over the cascade's triangles.

#include "Voxelize.hlsli"

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID, uint3 groups : SV_GroupID)
{
    CascadeData c = LoadCascade();
    RWByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    RWStructuredBuffer<uint2> items = ResourceDescriptorHeap[g.workItems];
    float inv = 1.0f / c.voxelSize;
    float3 origin = float3(Origin(c));
    int3 keep = int3(g.keepX, g.keepY, g.keepZ) / int(kVoxelBrick);
    uint stride = min((g.triangleCount + 63u) / 64u, kGroupsMax) * 64u;
    for (uint t = id.x; t < g.triangleCount; t += stride)
    {
        Triangle tri = LoadTriangle(t);
        // In the cascade's voxels, relative to its origin.
        float3 a = tri.p0 * inv - origin;
        float3 b = tri.p1 * inv - origin;
        float3 cc = tri.p2 * inv - origin;
        float3 triLo = min(a, min(b, cc));
        float3 triHi = max(a, max(b, cc));
        float3 lo = triLo - kVoxelBand;
        float3 hi = triHi + kVoxelBand;
        if (any(hi < 0.0f) || any(lo > float(kVoxelResolution)))
        {
            continue;
        }
        float3 n = cross(b - a, cc - a);
        float area2 = length(n);
        n = area2 > 1e-20f ? n / area2 : float3(0.0f, 1.0f, 0.0f);
        // Half a brick's extent along the normal: a brick lies that much closer to the plane than its centre.
        float extent = 0.5f * float(kVoxelBrick) * (abs(n.x) + abs(n.y) + abs(n.z));
        int3 b0 = clamp(int3(floor(lo / float(kVoxelBrick))), 0, int(kBrickResolution) - 1);
        int3 b1 = clamp(int3(floor(hi / float(kVoxelBrick))), 0, int(kBrickResolution) - 1);
        for (int z = b0.z; z <= b1.z; ++z)
        {
            for (int y = b0.y; y <= b1.y; ++y)
            {
                for (int x = b0.x; x <= b1.x; ++x)
                {
                    int3 brick = int3(x, y, z);
                    if (g.incremental != 0u && all(brick >= keep) && all(brick < keep + int(kBrickResolution)))
                    {
                        continue;
                    }
                    float3 boxLo = float3(brick) * float(kVoxelBrick);
                    float3 boxHi = boxLo + float(kVoxelBrick);
                    float plane = abs(dot(boxLo + 0.5f * float(kVoxelBrick) - a, n)) - extent;
                    if (plane > kVoxelBand)
                    {
                        continue;
                    }
                    bool near = plane <= kSurfaceDistance && all(boxHi >= triLo - kSurfaceDistance) &&
                                all(boxLo <= triHi + kSurfaceDistance);
                    uint slot;
                    counters.InterlockedAdd(kCounterWorkItems, 1u, slot);
                    if (slot < g.workCapacity)
                    {
                        items[slot] = uint2(t, PackBrick(uint3(brick)) | (near ? kWorkNear : 0u));
                    }
                    else
                    {
                        counters.InterlockedAdd(kCounterWorkOverflow, 1u);
                    }
                }
            }
        }
    }
}
