// air-shader-target: cs_6_6
// The light in and around the volumes for their ray march (docs/architecture.md §8.13): kVolumeProbes
// probes per volume on a lattice over its box, each tracing kVolumeProbeRays paths a frame (a thread
// group holds kProbesPerGroup probes, a thread per ray). They are the path tracer's paths through the volumes (tracked, AIR_PT_VOLUME), with kFlagProbe:
// without the sunlight and the light of the march's light groups that reach the probe without
// scattering, which the march adds itself. So a probe holds the rest of the light arriving at its
// point: the sky and the surfaces as the volumes dim them, and the light the volumes scatter, which
// the march's single scattering of it turns into all the orders of scattering beyond the first. Their
// radiance is projected onto L1 spherical harmonics and blended into the probe field
// (VolumeProbeConstants::blend), with the volume's occlusion at the probe (VolumeMarch.hlsli).

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<VolumeProbeConstants> g : register(b0);

#include "SceneAccess.hlsli"

HitInfo TraceClosest(float3 origin, float3 direction, uint bounce)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed())
    {
        if (CandidateIsSolid(query.CandidateType(), query.CandidateInstanceID(), query.CandidatePrimitiveIndex(),
                             query.CandidateTriangleBarycentrics()))
        {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    HitInfo hit;
    hit.hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    hit.t = hit.hit ? query.CommittedRayT() : -1.0f;
    hit.instance = hit.hit ? query.CommittedInstanceID() : 0u;
    hit.primitive = hit.hit ? query.CommittedPrimitiveIndex() : 0u;
    hit.barycentrics = hit.hit ? query.CommittedTriangleBarycentrics() : float2(0.0f, 0.0f);
    return hit;
}

float VolumeAoSample(VolumeMarchData v, float3 uvw)
{
    Texture3D<float> ao = ResourceDescriptorHeap[NonUniformResourceIndex(v.ao)];
    return ao.SampleLevel(gLinearClamp, uvw, 0.0f);
}

#define AIR_PT_VOLUME 1
#define AIR_PT_PROBE 1
#include "VolumeShadowHooks.hlsli"
#include "Integrator.hlsli"
#include "VolumeProbes.hlsli"

// The volume's occlusion at x, as VolumeMarch.hlsli reads it.
float ProbeOcclusion(VolumeMarchData v, float3 x)
{
    float3 uvw = (x - v.aoCorner) / (v.aoCell * float3(float(v.aoX), float(v.aoY), float(v.aoZ)));
    return VolumeAoSample(v, uvw);
}

groupshared float4 gProjected[kProbesPerGroup * kVolumeProbeRays][3];

[numthreads(kProbesPerGroup * kVolumeProbeRays, 1, 1)]
void Main(uint3 group : SV_GroupID, uint index : SV_GroupIndex)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    gVolumeShadows = f.volumeShadows;
    uint slot = index / kVolumeProbeRays; // the thread's probe in the group
    uint ray = index % kVolumeProbeRays;
    uint probe = group.x * kProbesPerGroup + slot; // kVolumeProbes is a multiple: one volume per group
    StructuredBuffer<VolumeMarchData> volumes = ResourceDescriptorHeap[g.volumes];
    VolumeMarchData v = volumes[probe / kVolumeProbes];

    IntegratorParams params;
    params.sky = f.sky;
    params.clouds = f.clouds;
    params.volumeCount = f.volumeCount;
    params.volumeTable = f.volumeTable;
    params.textureSets = f.textureSets;
    gTextureSets = f.textureSets;
    params.textureSeed = f.frameIndex;
    params.coatModel = f.coatModel;
    gNeuralCoat = f.neuralCoat;
    params.lightCount = f.lightCount;
    params.maxBounces = f.maxBounces;
    params.flags = (f.flags & (kFlagFurnace | kFlagNoNee)) | kFlagProbe;
    params.lightTreeBounces = f.lightTreeBounces;
    params.cloudEvents = f.cloudEvents;
    params.volumeEvents = f.volumeEvents;
    params.nrcSpread = 0.0f;
    params.nrcSlot = 0u;
    params.nrcRecords = 0u;

    uint local = probe % kVolumeProbes;
    float3 position = ProbePosition(v, local);
    float3 direction = ProbeRayDirection(probe, g.frame, ray);
    uint seed = Pcg(ray + Pcg(probe + Pcg(g.frame + 0x50524F42u)));
    PathSample s = TracePath(position, direction, params, seed);
    float3 radiance = all(isfinite(s.radiance)) ? s.radiance : float3(0.0f, 0.0f, 0.0f);
    float4 basis = ProbeBasis(direction) * (4.0f * kPi / float(kVolumeProbeRays));
    gProjected[index][0] = basis * radiance.x;
    gProjected[index][1] = basis * radiance.y;
    gProjected[index][2] = basis * radiance.z;
    GroupMemoryBarrierWithGroupSync();
    if (ray < kProbeBlocks)
    {
        // The probe's first threads write its blocks: thread b sums block b over the probe's rays.
        float4 value = float4(ProbeOcclusion(v, position), 0.0f, 0.0f, 0.0f);
        if (ray < 3u)
        {
            value = float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (uint i = 0u; i < kVolumeProbeRays; ++i)
            {
                value += gProjected[slot * kVolumeProbeRays + i][ray];
            }
        }
        RWTexture3D<float4> field = ResourceDescriptorHeap[g.probes];
        uint volume = probe / kVolumeProbes;
        uint3 texel = uint3(ProbeTexelX(volume, local), ProbeTexelY(local), ProbeTexelZ(local, ray));
        field[texel] = g.blend >= 1.0f ? value : lerp(field[texel], value, g.blend);
    }
}
