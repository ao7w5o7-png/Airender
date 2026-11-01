// Probe GI: voxelizing the scene's triangles into a cascade (ProbeGiShared.h), all of it or, where it moved,
// the bricks it did not cover before (the others are kept: VoxelizeConstants::incremental). The passes:
//   VoxelizeClear       distances to infinity and the voxel index to empty where voxelized again (their
//                       slots to the ring of free ones), the counters of the update to 0
//   VoxelizeBin         per triangle near the cascade: a work item per brick its band reaches (flagged
//                       kWorkNear where it may cross the brick's voxels)
//   VoxelizeDistance    per work item (64 threads, one per voxel of the brick): the exact distance to the
//                       triangle into the distance texture (atomic minimum); voxels it crosses are marked
//   VoxelizeAllocate    a slot for every marked voxel, free ones first
//   VoxelizeAttributes  per near work item: the triangle's albedo and normal, weighted, into the faces it
//                       shows of the voxels it crosses
//   VoxelizeResolve     per new surface voxel: the averages into a SurfaceVoxel, its first light
//   VoxelizeFinalize    per voxel: the distance into the R8 field, the bricks a surface crosses
//   VoxelizeBricks      x3, per axis: the Chebyshev distance in bricks to the nearest such brick
#ifndef AIR_PROBE_GI_VOXELIZE_HLSLI
#define AIR_PROBE_GI_VOXELIZE_HLSLI

#include "GiCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<VoxelizeConstants> g : register(b0);

// Distances to triangles count up to here (voxels): the band plus the brick a work item covers.
static const float kSurfaceDistance = 0.8660254f; // half a voxel's diagonal: the triangle may cross the voxel
static const uint kGroupsMax = 16384u;            // grid-stride dispatches use at most this many groups

struct Triangle
{
    float3 p0; // world, metres
    float3 p1;
    float3 p2;
    float3 n0; // world, unit
    float3 n1;
    float3 n2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
    uint material;
};

// The range holding global triangle t (binary search over firstTriangle).
VoxelizeRange FindRange(uint t)
{
    ByteAddressBuffer ranges = ResourceDescriptorHeap[g.ranges];
    uint lo = 0u;
    uint hi = g.rangeCount - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1u) >> 1;
        if (ranges.Load<VoxelizeRange>(g.rangeOffset + mid * 16u).firstTriangle <= t)
        {
            lo = mid;
        }
        else
        {
            hi = mid - 1u;
        }
    }
    return ranges.Load<VoxelizeRange>(g.rangeOffset + lo * 16u); // sizeof(VoxelizeRange)
}

float3 TransformPoint(InstanceData d, float3 p)
{
    float4 h = float4(p, 1.0f);
    return float3(dot(d.objectToWorld0, h), dot(d.objectToWorld1, h), dot(d.objectToWorld2, h));
}

float3 TransformNormal(InstanceData d, float3 n)
{
    float4 h = float4(n, 0.0f);
    return normalize(float3(dot(d.normalToWorld0, h), dot(d.normalToWorld1, h), dot(d.normalToWorld2, h)));
}

Triangle LoadTriangle(uint t)
{
    VoxelizeRange range = FindRange(t);
    StructuredBuffer<InstanceData> instances = ResourceDescriptorHeap[g.instances];
    InstanceData d = instances[range.instance];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(d.indexBuffer)];
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(d.vertexBuffer)];
    uint primitive = t - range.firstTriangle;
    Vertex v0 = vertices[indices[primitive * 3u]];
    Vertex v1 = vertices[indices[primitive * 3u + 1u]];
    Vertex v2 = vertices[indices[primitive * 3u + 2u]];
    Triangle tri;
    tri.p0 = TransformPoint(d, v0.position);
    tri.p1 = TransformPoint(d, v1.position);
    tri.p2 = TransformPoint(d, v2.position);
    tri.n0 = TransformNormal(d, v0.normal);
    tri.n1 = TransformNormal(d, v1.normal);
    tri.n2 = TransformNormal(d, v2.normal);
    tri.uv0 = float2(v0.u, v0.v);
    tri.uv1 = float2(v1.u, v1.v);
    tri.uv2 = float2(v2.u, v2.v);
    tri.material = d.materialIndex;
    return tri;
}

// The point of triangle (a, b, c) closest to p, as barycentrics (Ericson, Real-Time Collision Detection
// 5.1.5). Returns weights of a, b, c.
float3 ClosestBarycentrics(float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        return float3(1.0f, 0.0f, 0.0f);
    }
    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        return float3(0.0f, 1.0f, 0.0f);
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        return float3(1.0f - v, v, 0.0f);
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        return float3(0.0f, 0.0f, 1.0f);
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        return float3(1.0f - w, 0.0f, w);
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return float3(0.0f, 1.0f - w, w);
    }
    float denom = 1.0f / max(va + vb + vc, 1e-30f);
    float v = vb * denom;
    float w = vc * denom;
    return float3(1.0f - v - w, v, w);
}

CascadeData LoadCascade()
{
    ByteAddressBuffer frame = ResourceDescriptorHeap[g.frameData];
    return frame.Load<CascadeData>(g.frameOffset + kGiFrameCascades + g.cascade * 80u); // sizeof(CascadeData)
}

int3 Origin(CascadeData c)
{
    return int3(c.originX, c.originY, c.originZ);
}

uint3 Storage(int3 v)
{
    return uint3(v & int(kVoxelResolution - 1u));
}

// A work item's brick: its storage index packs the brick's coordinates relative to the cascade origin.
uint PackBrick(uint3 local)
{
    return local.x | (local.y << 5) | (local.z << 10);
}

uint3 UnpackBrick(uint packed)
{
    return uint3(packed & 31u, (packed >> 5) & 31u, (packed >> 10) & 31u);
}

// Whether the voxel at `local` (relative to the cascade's origin) keeps what it holds in this update.
bool Kept(int3 local)
{
    int3 keep = int3(g.keepX, g.keepY, g.keepZ);
    return g.incremental != 0u && all(local >= keep) && all(local < keep + int(kVoxelResolution));
}

// Storage coordinates -> coordinates relative to the cascade's origin.
int3 LocalOf(uint3 storage, int3 origin)
{
    return int3((storage - uint3(origin & int(kVoxelResolution - 1u))) & (kVoxelResolution - 1u));
}

// The colour and metallic (or a cutout's alpha) of a material's GI texture at uv, at the level whose
// texels are about `footprint` of the texture's [0, 1) wide.
float4 GiTextureSample(uint set, float2 uv, float footprint)
{
    StructuredBuffer<uint> table = ResourceDescriptorHeap[g.giTextures];
    uint index = table[set];
    if (index == kInvalidIndex)
    {
        return float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    Texture2D<float4> t = ResourceDescriptorHeap[NonUniformResourceIndex(index)];
    uint w;
    uint h;
    uint levels;
    t.GetDimensions(0u, w, h, levels);
    float level = log2(max(footprint * float(max(w, h)), 1.0f));
    return t.SampleLevel(gLinearWrap, uv, min(level, float(levels - 1u)));
}

// The surface's albedo as the voxels keep it: what the material reflects in all, averaged over directions
// (a dielectric's coat about 0.08 plus its base under it, a metal its average Fresnel), as if Lambertian.
// a = whether the point is solid (cutouts).
float4 VoxelMaterial(MaterialData m, float2 uv, float footprint)
{
    if (g.furnace != 0u)
    {
        return float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    float3 base = m.baseColor;
    float metallic = m.metallic;
    float solid = 1.0f;
    if (m.textureSet != kInvalidIndex)
    {
        float4 t = GiTextureSample(m.textureSet, uv * m.uvScale, footprint * m.uvScale);
        base *= t.xyz;
        if (m.alphaCutoff > 0.0f)
        {
            solid = t.w >= m.alphaCutoff ? 1.0f : 0.0f;
        }
        else
        {
            metallic *= t.w;
        }
    }
    float3 dielectric = 0.08f + 0.92f * base;
    float3 metal = base + (1.0f - base) / 21.0f;
    return float4(lerp(dielectric, metal, saturate(metallic)), solid);
}

#endif
