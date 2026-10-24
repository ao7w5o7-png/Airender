// air-shader-target: cs_6_6
// The volumes' moment shadow maps (docs/architecture.md §8.13): a thread per texel of every map (one
// map per dispatch slice), its ray marched from the light through every volume (VolumeShadowMarch.hlsli)
// and traced against the scene's geometry for its first opaque hit.

#include "air/renderer/shader/SceneShared.h"
#include "VolumeMarchShared.h"

[[vk::push_constant]] ConstantBuffer<VolumeShadowConstants> g : register(b0);

uint VolumeIndirectionLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.indirection)];
    return buffer[index];
}

uint VolumeOccupancyLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.occupancy)];
    return buffer[index];
}

uint VolumeBrickWordLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.bricks)];
    return buffer[index];
}

uint VolumeRangeLoad(VolumeBrickLevel l, uint brick)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.ranges)];
    return buffer[brick];
}

VolumeMarchData LoadMarchVolume(uint index)
{
    StructuredBuffer<VolumeMarchData> volumes = ResourceDescriptorHeap[g.volumes];
    return volumes[index];
}

VolumeBrickLevel LoadVolumeLevel(uint index)
{
    StructuredBuffer<VolumeBrickLevel> levels = ResourceDescriptorHeap[g.levels];
    return levels[index];
}

// MomentShadow.hlsli's lookups are not used here.
float MomentMapB0Sample(MomentShadowMap m, float2 uv)
{
    return 0.0f;
}

float4 MomentMapBSample(MomentShadowMap m, float2 uv)
{
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

float MomentMapOccluderLoad(MomentShadowMap m, uint x, uint y)
{
    return kMomentNoOccluder;
}

#include "air/renderer/shader/MomentShadow.hlsli"
#include "air/renderer/shader/VolumeBricks.hlsli"
#include "VolumeShadowMarch.hlsli"

// The first opaque hit along o + t d within (tMin, tMax), past any triangle of instance `skip`; tMax
// when there is none.
float FirstOpaqueHit(float3 o, float3 d, float tMin, float tMax, uint skip)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = o;
    ray.Direction = d;
    ray.TMin = tMin;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && query.CandidateInstanceID() != skip)
        {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : tMax;
}

// The depth of the texel ray's first opaque hit, kMomentNoOccluder without one; for the sun also
// kMomentOccluded when the geometry shades the ray before it reaches the near plane.
float OpaqueDepth(MomentShadowMap m, float2 uv)
{
    float3 o;
    float3 d;
    MomentMapRay(m, uv, o, d);
    float reach = m.depthFar + 0.5f * (m.depthFar - m.depthNear);
    if (m.kind == kMomentMapSun && FirstOpaqueHit(o, -d, 0.0f, 1e30f, kInvalidIndex) < 1e30f)
    {
        return kMomentOccluded;
    }
    float t = FirstOpaqueHit(o, d, m.kind == kMomentMapSun ? 0.0f : 1e-4f, reach, m.skipInstance);
    return t < reach ? MomentDepth(m, t) : kMomentNoOccluder;
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.size || id.y >= g.size)
    {
        return;
    }
    StructuredBuffer<MomentShadowMap> maps = ResourceDescriptorHeap[g.maps];
    MomentShadowMap m = maps[id.z];
    float2 uv = (float2(id.xy) + 0.5f) / float(g.size);
    float b0;
    float4 b;
    MarchShadowTexel(m, uv, g.volumeCount, g.level, g.stepScale, b0, b);
    RWTexture2DArray<float> b0Out = ResourceDescriptorHeap[g.b0];
    RWTexture2DArray<float4> bOut = ResourceDescriptorHeap[g.moments];
    RWTexture2DArray<float> occluderOut = ResourceDescriptorHeap[g.occluders];
    b0Out[uint3(id.xy, m.slice)] = b0;
    bOut[uint3(id.xy, m.slice)] = b;
    occluderOut[uint3(id.xy, m.slice)] = OpaqueDepth(m, uv);
}
