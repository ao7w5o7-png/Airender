// The real-time volumes' ray march (docs/architecture.md §8.13), shared by the GPU's march pass
// (VolumeMarch.hlsl) and its CPU twin (VolumeMarchReference.cpp): a camera ray through the volumes in
// front of its surface, in steps of a fraction of their voxels from a jittered start, every step
// scattering towards the camera the light of the sun and of a few light groups that reaches it
// through their moment shadow maps (MomentShadow.hlsli), and the rest of the light arriving there as
// the volume's probes hold it (VolumeProbes.hlsl: the sky and the surfaces through the volumes, and
// the light the volumes scatter, so that the orders of scattering beyond the first come from them).
// What the volumes scatter onto surfaces is not followed. Mueller describes this design for a production renderer (SIGGRAPH 2026):
// per-pixel marching instead of particles, shadows rendered once per light and looked up everywhere,
// ray segments by the volumes they cross.
//
// The including code defines, before including this file, VolumeBricks.hlsli's and MomentShadow.hlsli's
// hooks and
//   MomentShadowMap   LoadMomentMap(uint index)       VolumeShadowsData::maps
//   uint              VolumeGroupMap(uint group)      its moment shadow map, kInvalidIndex: none
//   LightGroup        LoadLightGroup(uint group)
//   LightTriangle     LoadLight(uint index)           the light list
//   uint              LoadMarchLight(uint i)          the light groups that light the volumes
//   VolumeMarchData   LoadMarchVolume(uint index)     GpuScene::volumeMarchTable
//   VolumeBrickLevel  LoadVolumeLevel(uint index)     GpuScene::volumeLevelTable
//   float             VolumeAoSample(VolumeMarchData v, float3 uvw)   its occlusion texture, trilinear
//   float4            ProbeFieldSample(float3 uvw)    the probe field (kProbeBlocks), trilinear
// and includes VolumeProbes.hlsli first.
#ifndef AIR_PT_VOLUME_MARCH_HLSLI
#define AIR_PT_VOLUME_MARCH_HLSLI

AIR_CONST float kMarchPi = 3.14159265358979f;
AIR_CONST float kMarchOpaque = 1e-3f; // transmittance at which a ray ends
AIR_CONST uint kMarchMaxSteps = 2048u;

struct MarchSetup
{
    SkyData sky;      // the sun's direction, radiance and disc
    uint sunMap;      // the sun's moment shadow map; kInvalidIndex: no sunlight in the volumes
    uint lightCount;  // light groups that light the volumes (LoadMarchLight)
    uint probeVolumes; // volumes in the probe field
    uint level;       // of the bricks
    float stepScale;  // the step, in voxels of that level
    uint flags;       // kMarchNoDirect, kMarchNoIndirect
};

// The volumes a ray crosses in front of its surface (at most kMarchMaxVolumes): their entries in the
// volume table (8 bits each) and where the ray is inside their boxes.
struct MarchRayVolumes
{
    uint count;
    uint indices;
    float4 tIn;
    float4 tOut;
};

// The volume's occlusion (its mean transmittance over all directions) at x.
float VolumeOcclusion(VolumeMarchData v, float3 x)
{
    float3 uvw = (x - v.aoCorner) / (v.aoCell * float3(float(v.aoX), float(v.aoY), float(v.aoZ)));
    return VolumeAoSample(v, uvw);
}

struct MarchResult
{
    float3 radiance;     // the light the volumes scatter towards the camera, nits
    float3 direct;       // its part from the sun and the light groups
    float transmittance; // what is left of the light from behind them
    float depth;         // the mean depth their light comes from (transmittance-weighted), for the air
    uint steps;          // lit steps
};

float Lane4(float4 v, uint k)
{
    return k == 0u ? v.x : (k == 1u ? v.y : (k == 2u ? v.z : v.w));
}

float4 SetLane4(float4 v, uint k, float value)
{
    return float4(k == 0u ? value : v.x, k == 1u ? value : v.y, k == 2u ? value : v.z, k == 3u ? value : v.w);
}

// Where a pixel's samples sit in their steps in a frame (0..1): its rank in the 64 x 64 blue noise,
// moved on by the golden ratio every frame (in 32-bit fixed point, the same on every processor), so
// that neighbouring pixels sample different depths and every pixel cycles through them evenly.
float MarchJitter(uint rank, uint frame)
{
    uint x = (rank << 20u) + (1u << 19u) + frame * 2654435769u;
    return float(x >> 8u) * (1.0f / 16777216.0f);
}

// Henyey-Greenstein; u is the cosine between the directions light travels in before and after.
float MarchPhase(float anisotropy, float u)
{
    float d = 1.0f + anisotropy * anisotropy - 2.0f * anisotropy * u;
    return (1.0f - anisotropy * anisotropy) / (4.0f * kMarchPi * d * sqrt(d));
}

// The phase functions of the volumes at a step weighted by their scattering coefficients.
float MixedPhase(float4 sigmaS, float4 phaseG, uint count, float u)
{
    float phase = 0.0f;
    for (uint k = 0u; k < count; ++k)
    {
        float s = Lane4(sigmaS, k);
        if (s > 0.0f)
        {
            phase += s * MarchPhase(Lane4(phaseG, k), u);
        }
    }
    return phase;
}

// The solid angle of the triangle (a, b, c) seen from the origin (Van Oosterom and Strackee 1983).
float TriangleSolidAngle(float3 a, float3 b, float3 c)
{
    float la = length(a);
    float lb = length(b);
    float lc = length(c);
    float numerator = abs(dot(a, cross(b, c)));
    float denominator = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
    return 2.0f * atan2(numerator, denominator);
}

// The light a group's emitters send through x towards the camera (ray direction d), per unit of the
// volumes' scattering (sigmaS, anisotropies phaseG), before its shadow: exactly for a few triangles (one-sided,
// each by its solid angle, the phase function at its centre), else as a sphere (a closed emitter) or
// a disc facing normalArea (an open one) of the group's area.
float3 GroupInScattering(LightGroup lg, float3 x, float3 d, float4 sigmaS, float4 phaseG, uint count)
{
    float3 result = float3(0.0f, 0.0f, 0.0f);
    if (lg.lightCount <= kExactGroupTriangles)
    {
        for (uint i = 0u; i < lg.lightCount; ++i)
        {
            LightTriangle tri = LoadLight(lg.firstLight + i);
            float3 a = tri.p0 - x;
            if (dot(cross(tri.e1, tri.e2), a) >= 0.0f)
            {
                continue; // x is behind the emitting side
            }
            float3 b = a + tri.e1;
            float3 c = a + tri.e2;
            float3 centre = a + (tri.e1 + tri.e2) * (1.0f / 3.0f);
            float u = dot(d, normalize(centre));
            result += tri.radiance * (TriangleSolidAngle(a, b, c) * MixedPhase(sigmaS, phaseG, count, u));
        }
        return result;
    }
    float3 toLight = lg.centre - x;
    float distance = max(length(toLight), 1e-6f);
    float3 w = toLight / distance;
    float solidAngle = 0.0f;
    float open = length(lg.normalArea);
    if (open < 0.5f * lg.area)
    {
        float r2 = lg.area / (4.0f * kMarchPi);
        solidAngle = 2.0f * kMarchPi * (1.0f - sqrt(max(0.0f, 1.0f - r2 / (distance * distance))));
    }
    else
    {
        float cosEmitter = dot(lg.normalArea / open, -w);
        float r2 = lg.area / kMarchPi;
        solidAngle = cosEmitter > 0.0f ? 2.0f * kMarchPi * (1.0f - distance / sqrt(distance * distance + r2)) * cosEmitter
                                       : 0.0f;
    }
    return lg.radiance * (solidAngle * MixedPhase(sigmaS, phaseG, count, dot(d, w)));
}

// The light a volume's probes hold (L1 spherical harmonics of the radiance arriving from each
// direction, one float4 per colour channel: Y00, Y1-1, Y10, Y11), scattered towards the camera along
// ray direction d by Henyey-Greenstein g (which scales band l by g^l).
float3 ProbeInScattering(float4 red, float4 green, float4 blue, float3 d, float anisotropy)
{
    float y1 = 0.488602512f * anisotropy;
    float4 basis = float4(0.282094792f, y1 * d.y, y1 * d.z, y1 * d.x);
    return float3(max(dot(red, basis), 0.0f), max(dot(green, basis), 0.0f), max(dot(blue, basis), 0.0f));
}

// Where the ray o + t d crosses volume v's box within [0, tMax].
bool MarchVolumeSegment(VolumeMarchData v, float3 o, float3 d, float tMax, AIR_OUT(float) tIn, AIR_OUT(float) tOut)
{
    return BoxSegment(o, d, v.boundsMin, v.boundsMax, 0.0f, tMax, tIn, tOut);
}

// Where a tile reads the light around a volume: on the ray through its centre, half way through the
// volume's box (or, when that ray misses it, as close to the box's centre as the ray comes, inside the box).
float3 TileProbePoint(VolumeMarchData v, float3 o, float3 d)
{
    float t0;
    float t1;
    if (BoxSegment(o, d, v.boundsMin, v.boundsMax, 0.0f, 1e30f, t0, t1))
    {
        return o + d * (0.5f * (t0 + t1));
    }
    float3 centre = (v.boundsMin + v.boundsMax) * 0.5f;
    float3 p = o + d * max(dot(centre - o, d), 0.0f);
    return min(max(p, v.boundsMin), v.boundsMax);
}

// The volumes' light along o + t d up to tSurface; jitter in [0, 1) places the samples in their steps.
MarchResult MarchRay(MarchSetup s, MarchRayVolumes rv, float3 o, float3 d, float tSurface, float jitter)
{
    MarchResult result;
    result.radiance = float3(0.0f, 0.0f, 0.0f);
    result.direct = float3(0.0f, 0.0f, 0.0f);
    result.transmittance = 1.0f;
    result.depth = 0.0f;
    result.steps = 0u;
    if (rv.count == 0u)
    {
        return result;
    }
    // From the first box the ray enters to the last it leaves (or its surface), in steps of a fraction
    // of the smallest voxel among its volumes.
    float tStart = 1e30f;
    float tEnd = 0.0f;
    float voxel = 1e30f;
    for (uint k = 0u; k < rv.count; ++k)
    {
        tStart = min(tStart, Lane4(rv.tIn, k));
        tEnd = max(tEnd, Lane4(rv.tOut, k));
        VolumeMarchData v = LoadMarchVolume((rv.indices >> (8u * k)) & 0xFFu);
        voxel = min(voxel, LoadVolumeLevel(v.levels + s.level).voxelSize);
    }
    tEnd = min(tEnd, tSurface);
    float step = s.stepScale * voxel;
    bool sun = (s.flags & kMarchNoDirect) == 0u && s.sunMap != kInvalidIndex;
    bool lights = (s.flags & (kMarchNoDirect | kMarchFurnace)) == 0u;
    bool furnace = (s.flags & kMarchFurnace) != 0u;
    bool ambient = (s.flags & kMarchNoIndirect) == 0u;
    float sunScale = sun ? SunSolidAngle(s.sky) : 0.0f;
    float sunCosine = dot(d, s.sky.sunDirection);
    float weightedDepth = 0.0f;
    float weight = 0.0f;
    for (uint i = 0u; i < kMarchMaxSteps; ++i)
    {
        float t0 = tStart + float(i) * step;
        if (t0 >= tEnd)
        {
            break;
        }
        float segment = min(step, tEnd - t0);
        float t = t0 + jitter * segment;
        float3 x = o + d * t;
        // The volumes the step is in, their extinction and scattering; where all of them are empty
        // around x, how far that emptiness reaches along the ray.
        float sigmaT = 0.0f;
        float4 sigmaS = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 phaseG = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ambientR = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ambientG = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ambientB = float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool occupied = false;
        float skipTo = tEnd;
        for (uint k = 0u; k < rv.count; ++k)
        {
            if (t < Lane4(rv.tIn, k))
            {
                skipTo = min(skipTo, Lane4(rv.tIn, k)); // not entered yet
                continue;
            }
            if (t >= Lane4(rv.tOut, k))
            {
                continue;
            }
            VolumeMarchData v = LoadMarchVolume((rv.indices >> (8u * k)) & 0xFFu);
            VolumeBrickLevel level = LoadVolumeLevel(v.levels + s.level);
            float3 emptyMin;
            float3 emptyMax;
            if (!BrickOccupied(level, x, emptyMin, emptyMax))
            {
                skipTo = min(skipTo, emptyMin.x == emptyMax.x ? t : t + BoxExit(x, d, emptyMin, emptyMax));
                continue;
            }
            occupied = true;
            float density = BrickDensity(level, x);
            if (density <= 0.0f)
            {
                skipTo = t;
                continue;
            }
            float extinction = v.extinction * density;
            sigmaT += extinction;
            sigmaS = SetLane4(sigmaS, k, furnace ? extinction : extinction * v.albedo);
            phaseG = SetLane4(phaseG, k, v.anisotropy);
            if (ambient)
            {
                // The probes' light at x, scaled by how much more or less the volume occludes x than
                // the probes it is filtered from (the occlusion grid is finer than the probes).
                uint volume = (rv.indices >> (8u * k)) & 0xFFu;
                float4 red = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 0u, x));
                float4 green = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 1u, x));
                float4 blue = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 2u, x));
                float probeOcclusion = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 3u, x)).x;
                float ratio = furnace ? 1.0f : min(VolumeOcclusion(v, x) / max(probeOcclusion, 0.05f), 4.0f);
                float3 around = ProbeInScattering(red, green, blue, d, v.anisotropy) * ratio;
                ambientR = SetLane4(ambientR, k, around.x);
                ambientG = SetLane4(ambientG, k, around.y);
                ambientB = SetLane4(ambientB, k, around.z);
            }
        }
        if (!occupied)
        {
            // Jump to the step that holds the end of the emptiness.
            if (skipTo > t0 + step)
            {
                i = uint(floor((skipTo - tStart) / step)) - 1u;
            }
            continue;
        }
        if (sigmaT <= 0.0f)
        {
            continue;
        }

        // The light scattered towards the camera at x per unit length: sigma_s * phase * incident light.
        float3 direct = float3(0.0f, 0.0f, 0.0f);
        if (sun)
        {
            float shadow = MomentLightVisibility(LoadMomentMap(s.sunMap), x);
            if (shadow > 0.0f)
            {
                direct += s.sky.sunRadiance * (sunScale * shadow * MixedPhase(sigmaS, phaseG, rv.count, sunCosine));
            }
        }
        if (lights)
        {
            for (uint j = 0u; j < s.lightCount; ++j)
            {
                uint group = LoadMarchLight(j);
                float shadow = MomentLightVisibility(LoadMomentMap(VolumeGroupMap(group)), x);
                if (shadow > 0.0f)
                {
                    direct += GroupInScattering(LoadLightGroup(group), x, d, sigmaS, phaseG, rv.count) * shadow;
                }
            }
        }
        float3 scattered = direct;
        if (ambient)
        {
            for (uint k = 0u; k < rv.count; ++k)
            {
                scattered += float3(Lane4(ambientR, k), Lane4(ambientG, k), Lane4(ambientB, k)) * Lane4(sigmaS, k);
            }
        }
        // Energy-conserving integration over the step (constant extinction and light along it).
        float stepTransmittance = exp(-sigmaT * segment);
        float integral = result.transmittance * (1.0f - stepTransmittance) / sigmaT;
        result.radiance += scattered * integral;
        result.direct += direct * integral;
        float absorbed = result.transmittance * (1.0f - stepTransmittance);
        weightedDepth += absorbed * t;
        weight += absorbed;
        result.transmittance *= stepTransmittance;
        ++result.steps;
        if (result.transmittance < kMarchOpaque)
        {
            break;
        }
    }
    result.depth = weight > 0.0f ? weightedDepth / weight : tStart;
    return result;
}

#endif
