// The volumes' probes (VolumeProbes.hlsl, docs/architecture.md §8.13), shared by C++ and HLSL: where
// they sit, which rays they trace, and where the probe field (kProbeBlocks, VolumeMarchShared.h) keeps
// and filters their light. Needs Sampling.hlsli (Pcg, Rand).
#ifndef AIR_PT_VOLUME_PROBES_HLSLI
#define AIR_PT_VOLUME_PROBES_HLSLI

// Probe i (0 .. kVolumeProbes - 1, x fastest) of a volume: the centres of a lattice of kVolumeProbeAxis^3
// cells over its box, so that none sits on the box's faces (often on a surface).
float3 ProbePosition(VolumeMarchData v, uint i)
{
    float n = float(kVolumeProbeAxis);
    uint x = i % kVolumeProbeAxis;
    uint y = (i / kVolumeProbeAxis) % kVolumeProbeAxis;
    uint z = i / (kVolumeProbeAxis * kVolumeProbeAxis);
    float3 cell = float3(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
    return v.boundsMin + (v.boundsMax - v.boundsMin) * (cell / n);
}

// Ray `index` of probe `probe` in frame `frame`: a Fibonacci lattice over the sphere (even coverage),
// turned by a random rotation per probe and frame (Shoemake's uniform quaternion).
float3 ProbeRayDirection(uint probe, uint frame, uint index)
{
    float z = 1.0f - (2.0f * float(index) + 1.0f) / float(kVolumeProbeRays);
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.39996323f * float(index); // the golden angle
    float3 p = float3(r * cos(phi), r * sin(phi), z);
    uint seed = Pcg(probe + Pcg(frame + 0x524F5441u));
    float u1 = Rand(seed);
    float u2 = Rand(seed);
    float u3 = Rand(seed);
    float a = sqrt(1.0f - u1);
    float b = sqrt(u1);
    float3 q = float3(a * sin(6.28318530718f * u2), a * cos(6.28318530718f * u2), b * sin(6.28318530718f * u3));
    float w = b * cos(6.28318530718f * u3);
    float3 c = cross(q, p);
    return p + (c * w + cross(q, c)) * 2.0f;
}

// L1 spherical harmonics of direction d: Y00, Y1-1, Y10, Y11.
float4 ProbeBasis(float3 d)
{
    return float4(0.282094792f, 0.488602512f * d.y, 0.488602512f * d.z, 0.488602512f * d.x);
}

// The probe field's texel of probe `probe` of the volume at `volume` in the table, in block `block`
// (x, y and z of the texel in that order).
uint ProbeTexelX(uint volume, uint probe)
{
    return volume * kVolumeProbeAxis + probe % kVolumeProbeAxis;
}

uint ProbeTexelY(uint probe)
{
    return (probe / kVolumeProbeAxis) % kVolumeProbeAxis;
}

uint ProbeTexelZ(uint probe, uint block)
{
    return block * kVolumeProbeAxis + probe / (kVolumeProbeAxis * kVolumeProbeAxis);
}

// Where the probe field filters block `block` of volume `volume` (of `volumes`) at x: trilinear between
// its probes, clamped to the outer ones, never reaching another volume's or block's texels.
float3 ProbeFieldUvw(VolumeMarchData v, uint volume, uint volumes, uint block, float3 x)
{
    float n = float(kVolumeProbeAxis);
    float3 f = (x - v.boundsMin) / max(v.boundsMax - v.boundsMin, float3(1e-6f, 1e-6f, 1e-6f)) * n;
    float lo = 0.5f;
    float hi = n - 0.5f;
    return float3((float(volume) * n + clamp(f.x, lo, hi)) / (n * float(volumes)), clamp(f.y, lo, hi) / n,
                  (float(block) * n + clamp(f.z, lo, hi)) / (n * float(kProbeBlocks)));
}

#endif
