// Probe GI screen probes: the directions of a probe's rays (ProbeGiShared.h), the same in every pass that traces or
// integrates them. A ray per cell of a kScreenProbeTexels^2 grid over the hemisphere around the probe's normal (the
// hemi-octahedral map, cells of near-equal solid angle), jittered within its cell by the probe's offset this frame.
#ifndef AIR_PROBE_GI_PROBE_RAYS_HLSLI
#define AIR_PROBE_GI_PROBE_RAYS_HLSLI

#include "GiCommon.hlsli"

// Hemi-octahedral map: the square [-1, 1]^2 onto the hemisphere z >= 0.
float3 HemiOctDecode(float2 e)
{
    float2 t = float2(e.x + e.y, e.x - e.y) * 0.5f;
    float3 v = float3(t, 1.0f - abs(t.x) - abs(t.y));
    return normalize(v);
}

struct ProbeRayJitter
{
    float2 cell; // offset within the cells
    float march; // the screen trace's step offset
};

ProbeRayJitter ProbeJitter(GiFrame f, uint probe)
{
    uint seed = Pcg(probe ^ Pcg(f.frameIndex));
    ProbeRayJitter j;
    j.cell = Rand2(seed);
    j.march = Rand(seed);
    return j;
}

float3 ProbeRayDirection(Basis basis, uint2 cell, float2 jitter)
{
    float2 e = (float2(cell) + jitter) / float(kScreenProbeTexels) * 2.0f - 1.0f;
    return ToWorld(basis, HemiOctDecode(e));
}

#endif
