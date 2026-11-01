// Probe GI voxelization (Voxelize.hlsli): per near work item (kWorkNear), for the voxels its triangle crosses: the
// albedo at the closest point (the material's GI texture at the voxel's footprint) and the shading normal,
// weighted by closeness and by how squarely each side faces each axis, added into the faces of the voxel
// that side shows. Both sides of a triangle reflect (the path tracer's surfaces do); a back side counts a
// quarter, so that a closed mesh's inner sides do not mix into a thin wall's outer ones.

#include "Voxelize.hlsli"

static const float kFaceThreshold = 0.2f; // a side shows on the faces whose axis it faces by more than this
static const float kBackSide = 0.25f;

void Accumulate(RWStructuredBuffer<uint> acc, uint base, float3 albedo, float3 normal, float offset, float w)
{
    float s = w * kAccumulatorScale;
    InterlockedAdd(acc[base + 0u], uint(albedo.x * s + 0.5f));
    InterlockedAdd(acc[base + 1u], uint(albedo.y * s + 0.5f));
    InterlockedAdd(acc[base + 2u], uint(albedo.z * s + 0.5f));
    InterlockedAdd(acc[base + 3u], uint(s + 0.5f));
    InterlockedAdd(acc[base + 4u], asuint(int(round(normal.x * s))));
    InterlockedAdd(acc[base + 5u], asuint(int(round(normal.y * s))));
    InterlockedAdd(acc[base + 6u], asuint(int(round(normal.z * s))));
    InterlockedAdd(acc[base + 7u], asuint(int(round(offset * s))));
}

[numthreads(64, 1, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    CascadeData c = LoadCascade();
    ByteAddressBuffer counters = ResourceDescriptorHeap[g.counters];
    StructuredBuffer<uint2> items = ResourceDescriptorHeap[g.workItems];
    RWTexture3D<uint> voxelIndex = ResourceDescriptorHeap[g.voxelIndex];
    RWStructuredBuffer<uint> acc = ResourceDescriptorHeap[g.accumulators];
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[g.materials];
    uint count = min(counters.Load(kCounterWorkItems), g.workCapacity);
    float inv = 1.0f / c.voxelSize;
    int3 origin = Origin(c);
    uint3 inBrick = uint3(thread.x & 3u, (thread.x >> 2) & 3u, thread.x >> 4);
    for (uint item = group.x; item < count; item += kGroupsMax)
    {
        uint2 work = items[item];
        if ((work.y & kWorkNear) == 0u)
        {
            continue;
        }
        Triangle tri = LoadTriangle(work.x);
        uint3 local = UnpackBrick(work.y) * kVoxelBrick + inBrick;
        int3 voxel = origin + int3(local);
        float3 p = (float3(voxel) + 0.5f) * c.voxelSize;
        float3 w = ClosestBarycentrics(p, tri.p0, tri.p1, tri.p2);
        float3 q = tri.p0 * w.x + tri.p1 * w.y + tri.p2 * w.z;
        float d = length(p - q) * inv;
        if (d >= kSurfaceDistance)
        {
            continue;
        }
        uint slot = voxelIndex[Storage(voxel)];
        if (slot == 0u || slot == kVoxelMarked)
        {
            continue;
        }
        --slot;
        MaterialData m = materials[tri.material];
        float2 uv = tri.uv0 * w.x + tri.uv1 * w.y + tri.uv2 * w.z;
        if (m.alphaCutoff > 0.0f && VoxelMaterial(m, uv, 0.0f).w <= 0.0f)
        {
            continue;
        }
        // The texture's footprint: a voxel's width in texture coordinates on this triangle.
        float worldArea = length(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
        float2 e1 = tri.uv1 - tri.uv0;
        float2 e2 = tri.uv2 - tri.uv0;
        float uvArea = abs(e1.x * e2.y - e1.y * e2.x);
        float footprint = worldArea > 1e-20f ? c.voxelSize * sqrt(uvArea / worldArea) : 0.0f;
        float3 albedo = VoxelMaterial(m, uv, footprint).xyz;
        float3 n = normalize(tri.n0 * w.x + tri.n1 * w.y + tri.n2 * w.z);
        // Weighted by how much of the voxel the triangle can cover (its area, at most a voxel's face): a dense
        // mesh's many small triangles (a lamp's 300) must not outweigh a large one's surface around them.
        float area = min(0.5f * worldArea * inv * inv, 1.0f);
        float closeness = (1.0f - d / kSurfaceDistance) * area;
        // Where the surface lies from the voxel's centre along its normal, in voxels (lighting is computed
        // there, not at the centre).
        float offset = dot(q - p, n) * inv;
        for (uint face = 0u; face < 6u; ++face)
        {
            float3 axis = float3(face < 2u ? 1.0f : 0.0f, face == 2u || face == 3u ? 1.0f : 0.0f, face >= 4u ? 1.0f : 0.0f);
            axis *= (face & 1u) != 0u ? -1.0f : 1.0f;
            float facing = dot(n, axis);
            uint base = slot * kAccumulatorStride + face * kAccumulatorFace;
            if (facing > kFaceThreshold)
            {
                Accumulate(acc, base, albedo, n, offset, closeness * facing);
            }
            else if (-facing > kFaceThreshold)
            {
                Accumulate(acc, base, albedo, -n, -offset, closeness * -facing * kBackSide);
            }
        }
    }
}
