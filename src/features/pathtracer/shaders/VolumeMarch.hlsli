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
//   uint              VolumeGroupMap(uint group)      its first moment shadow map, kInvalidIndex: none
//   LightGroup        LoadLightGroup(uint group)
//   LightTriangle     LoadLight(uint index)           the light list
//   uint              LoadMarchLight(uint i)          the light groups that light the volumes
//   VolumeMarchData   LoadRayVolume(uint volume, uint slot)   entry `volume` of GpuScene::volumeMarchTable
//   VolumeBrickLevel  LoadRayLevel(uint volume, uint slot, uint level)   its level `level` (volumeLevelTable),
//                                                     both as a ray's tile keeps them in slot `slot`
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
    SkyData sky;       // the sun's direction, radiance and disc
    uint sunMap;       // the sun's moment shadow map; kInvalidIndex: no sunlight in the volumes
    uint lightCount;   // light groups that light the volumes (LoadMarchLight)
    uint probeVolumes; // volumes in the probe field
    uint level;        // of the bricks
    float stepScale;   // the step, in voxels of that level
    uint flags;        // kMarchNoDirect, kMarchNoIndirect
    uint lightSample;  // which of a light's maps the ray reads (MarchLightSample)
    uint lightMaps0;   // the map the ray reads of each light group, 8 bits each: groups 0..3
    uint lightMaps1;   // groups 4..7 (MarchChooseMaps)
};

// Which of a light's maps (when it has several, VolumeShadowLayout.h) a pixel reads in a frame: its
// quarter of the 64 x 64 blue noise ranks, moved on by one every frame, so that neighbouring pixels read
// different maps and every pixel reads each of them in turn (accumulation and TAA average them).
uint MarchLightSample(uint rank, uint frame)
{
    return (rank >> 10u) + frame;
}

// Chooses once per ray the map of each light group that its steps read (map lightSample of the
// group's maps), into s.lightMaps0 and s.lightMaps1.
void MarchChooseMaps(AIR_INOUT(MarchSetup) s)
{
    s.lightMaps0 = 0u;
    s.lightMaps1 = 0u;
    for (uint j = 0u; j < s.lightCount; ++j)
    {
        uint first = VolumeGroupMap(LoadMarchLight(j));
        uint count = LoadMomentMap(first).count;
        uint map = count > 1u ? first + s.lightSample % count : first;
        if (j < 4u)
        {
            s.lightMaps0 |= map << (8u * j);
        }
        else
        {
            s.lightMaps1 |= map << (8u * (j - 4u));
        }
    }
}

uint MarchLightMap(MarchSetup s, uint j)
{
    return ((j < 4u ? s.lightMaps0 : s.lightMaps1) >> (8u * (j & 3u))) & 0xFFu;
}

// The volumes a ray crosses in front of its surface (at most kMarchMaxVolumes): their entries in the
// volume table and their slots among its tile's volumes (8 bits each), and where the ray is inside
// their boxes.
struct MarchRayVolumes
{
    uint count;
    uint indices;
    uint slots;
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

// The volumes at point x (t along the ray, before tEnd): their extinction, their scattering by
// anisotropy (sigmaS, phaseG: volumes of the same anisotropy share a lane, so that MixedPhase evaluates
// the phase function once for them), the light their probes hold that they scatter towards the camera
// per unit length (with `ambient`), and, where they are all empty around x, how far that emptiness
// reaches.
struct MarchPoint
{
    float sigmaT;
    float4 sigmaS;
    float4 phaseG;
    float3 around;
    bool occupied;
    float skipTo;
};

MarchPoint NoPoint(float skipTo)
{
    MarchPoint p;
    p.sigmaT = 0.0f;
    p.sigmaS = float4(0.0f, 0.0f, 0.0f, 0.0f);
    p.phaseG = float4(0.0f, 0.0f, 0.0f, 0.0f);
    p.around = float3(0.0f, 0.0f, 0.0f);
    p.occupied = false;
    p.skipTo = skipTo;
    return p;
}

// Volume k's scattering at a point, added to the lane of its anisotropy.
void AddScattering(AIR_INOUT(MarchPoint) p, uint k, float scattering, float anisotropy)
{
    uint lane = k;
    for (uint m = 0u; m < k; ++m)
    {
        if (lane == k && Lane4(p.sigmaS, m) > 0.0f && Lane4(p.phaseG, m) == anisotropy)
        {
            lane = m;
        }
    }
    p.sigmaS = SetLane4(p.sigmaS, lane, Lane4(p.sigmaS, lane) + scattering);
    p.phaseG = SetLane4(p.phaseG, lane, anisotropy);
}

// The light volume v's probes hold at x that it scatters towards the camera per unit of its
// scattering, scaled by how much more or less the volume occludes x than the probes it is filtered
// from (the occlusion grid is finer than the probes).
float3 ProbeAround(MarchSetup s, VolumeMarchData v, uint volume, float3 x, float3 d)
{
    float4 red = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 0u, x));
    float4 green = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 1u, x));
    float4 blue = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 2u, x));
    float probeOcclusion = ProbeFieldSample(ProbeFieldUvw(v, volume, s.probeVolumes, 3u, x)).x;
    bool furnace = (s.flags & kMarchFurnace) != 0u;
    float ratio = furnace ? 1.0f : min(VolumeOcclusion(v, x) / max(probeOcclusion, 0.05f), 4.0f);
    return ProbeInScattering(red, green, blue, d, v.anisotropy) * ratio;
}

MarchPoint MarchPointAt(MarchSetup s, MarchRayVolumes rv, float3 x, float3 d, float t, float tEnd, bool ambient)
{
    MarchPoint p = NoPoint(tEnd);
    bool furnace = (s.flags & kMarchFurnace) != 0u;
    for (uint k = 0u; k < rv.count; ++k)
    {
        if (t < Lane4(rv.tIn, k))
        {
            p.skipTo = min(p.skipTo, Lane4(rv.tIn, k)); // not entered yet
            continue;
        }
        if (t >= Lane4(rv.tOut, k))
        {
            continue;
        }
        uint volume = (rv.indices >> (8u * k)) & 0xFFu;
        uint slot = (rv.slots >> (8u * k)) & 0xFFu;
        bool occupied;
        float3 emptyMin;
        float3 emptyMax;
        float density = BrickSample(LoadRayLevel(volume, slot, s.level), x, occupied, emptyMin, emptyMax);
        if (!occupied)
        {
            p.skipTo = min(p.skipTo, emptyMin.x == emptyMax.x ? t : t + BoxExit(x, d, emptyMin, emptyMax));
            continue;
        }
        p.occupied = true;
        if (density <= 0.0f)
        {
            p.skipTo = t;
            continue;
        }
        VolumeMarchData v = LoadRayVolume(volume, slot);
        float extinction = v.extinction * density;
        float scattering = furnace ? extinction : extinction * v.albedo;
        p.sigmaT += extinction;
        AddScattering(p, k, scattering, v.anisotropy);
        if (ambient)
        {
            p.around += ProbeAround(s, v, volume, x, d) * scattering;
        }
    }
    return p;
}

// How much of a light group's light reaches x in the volumes, from map `sample` of its maps (points
// spread over a large emitter close to them, VolumeShadowLayout.h): their mean over the frames.
float GroupVisibility(uint first, float3 x, uint sample)
{
    MomentShadowMap m = LoadMomentMap(first);
    if (m.count > 1u)
    {
        m = LoadMomentMap(first + sample % m.count);
    }
    return MomentLightVisibility(m, x);
}

// The light of the sun and of the light groups that the volumes at point p scatter towards the camera
// at x, per unit length. Each light group's light comes through the map the ray chose
// (MarchChooseMaps), which GroupVisibility(first, x, s.lightSample) reads.
float3 MarchDirect(MarchSetup s, MarchRayVolumes rv, MarchPoint p, float3 x, float3 d)
{
    float3 direct = float3(0.0f, 0.0f, 0.0f);
    if ((s.flags & kMarchNoDirect) != 0u)
    {
        return direct;
    }
    if (s.sunMap != kInvalidIndex)
    {
        float shadow = MomentLightVisibility(LoadMomentMap(s.sunMap), x);
        if (shadow > 0.0f)
        {
            float phase = MixedPhase(p.sigmaS, p.phaseG, rv.count, dot(d, s.sky.sunDirection));
            direct += s.sky.sunRadiance * (SunSolidAngle(s.sky) * shadow * phase);
        }
    }
    if ((s.flags & kMarchFurnace) == 0u)
    {
        for (uint j = 0u; j < s.lightCount; ++j)
        {
            float shadow = MomentLightVisibility(LoadMomentMap(MarchLightMap(s, j)), x);
            if (shadow > 0.0f)
            {
                direct += GroupInScattering(LoadLightGroup(LoadMarchLight(j)), x, d, p.sigmaS, p.phaseG, rv.count) *
                          shadow;
            }
        }
    }
    return direct;
}

// The light the volumes at x (t along the ray) would scatter towards the camera per unit of their
// scattering: their phase functions and probes mixed by their scattering there, the empty ones counted
// alike. valid: x lies in one of their boxes. What a pixel lights for its neighbours under adaptive
// sampling (VolumeMarch.hlsl), where the density at x need not be its neighbours'.
float3 MarchLightAt(MarchSetup s, MarchRayVolumes rv, float3 x, float3 d, float t, bool ambient, AIR_OUT(bool) valid)
{
    MarchPoint p = NoPoint(t);
    bool furnace = (s.flags & kMarchFurnace) != 0u;
    float total = 0.0f;
    for (uint k = 0u; k < rv.count; ++k)
    {
        if (t < Lane4(rv.tIn, k) || t >= Lane4(rv.tOut, k))
        {
            continue;
        }
        uint volume = (rv.indices >> (8u * k)) & 0xFFu;
        uint slot = (rv.slots >> (8u * k)) & 0xFFu;
        VolumeMarchData v = LoadRayVolume(volume, slot);
        float extinction = v.extinction * BrickDensity(LoadRayLevel(volume, slot, s.level), x);
        float weight = (furnace ? extinction : extinction * v.albedo) + 1e-6f;
        AddScattering(p, k, weight, v.anisotropy);
        if (ambient)
        {
            p.around += ProbeAround(s, v, volume, x, d) * weight;
        }
        total += weight;
    }
    valid = total > 0.0f;
    return valid ? (MarchDirect(s, rv, p, x, d) + p.around) / total : float3(0.0f, 0.0f, 0.0f);
}

// Where a ray marches: from the first box it enters to the last it leaves (or its surface), in steps
// of a fraction of the smallest voxel among its volumes.
void MarchSpan(MarchSetup s, MarchRayVolumes rv, float tSurface, AIR_OUT(float) tStart, AIR_OUT(float) tEnd,
               AIR_OUT(float) step)
{
    tStart = 1e30f;
    tEnd = 0.0f;
    float voxel = 1e30f;
    for (uint k = 0u; k < rv.count; ++k)
    {
        tStart = min(tStart, Lane4(rv.tIn, k));
        tEnd = max(tEnd, Lane4(rv.tOut, k));
        voxel = min(voxel, LoadRayLevel((rv.indices >> (8u * k)) & 0xFFu, (rv.slots >> (8u * k)) & 0xFFu, s.level).voxelSize);
    }
    tEnd = min(tEnd, tSurface);
    step = s.stepScale * voxel;
}

// One step of the march: the light scattered over [t0, t0 + segment] (`scattered` per unit length,
// sampled at t) and the extinction there, integrated as if constant over the step (energy conserving).
void MarchIntegrate(AIR_INOUT(MarchResult) result, AIR_INOUT(float) weightedDepth, AIR_INOUT(float) weight,
                    float3 scattered, float sigmaT, float segment, float t)
{
    float stepTransmittance = exp(-sigmaT * segment);
    float absorbed = result.transmittance * (1.0f - stepTransmittance);
    result.radiance += scattered * (absorbed / sigmaT);
    weightedDepth += absorbed * t;
    weight += absorbed;
    result.transmittance *= stepTransmittance;
    ++result.steps;
}

MarchResult NoMarch()
{
    MarchResult result;
    result.radiance = float3(0.0f, 0.0f, 0.0f);
    result.transmittance = 1.0f;
    result.depth = 0.0f;
    result.steps = 0u;
    return result;
}

// Step i of the march along o + t d (i moves on past emptiness): false once the ray is done.
bool MarchStep(MarchSetup s, MarchRayVolumes rv, float3 o, float3 d, float tStart, float tEnd, float step,
               float jitter, bool ambient, AIR_INOUT(uint) i, AIR_INOUT(MarchResult) result,
               AIR_INOUT(float) weightedDepth, AIR_INOUT(float) weight)
{
    float t0 = tStart + float(i) * step;
    if (t0 >= tEnd)
    {
        return false;
    }
    float segment = min(step, tEnd - t0);
    float t = t0 + jitter * segment;
    float3 x = o + d * t;
    MarchPoint p = MarchPointAt(s, rv, x, d, t, tEnd, ambient);
    ++i;
    if (!p.occupied)
    {
        // On to the step that holds the end of the emptiness.
        if (p.skipTo > t0 + step)
        {
            i = uint(floor((p.skipTo - tStart) / step));
        }
        return true;
    }
    if (p.sigmaT <= 0.0f)
    {
        return true;
    }
    MarchIntegrate(result, weightedDepth, weight, MarchDirect(s, rv, p, x, d) + p.around, p.sigmaT, segment, t);
    return result.transmittance >= kMarchOpaque;
}

// The volumes' light along o + t d up to tSurface, every step lit on its own; jitter in [0, 1) places
// the samples in their steps.
MarchResult MarchRay(MarchSetup s, MarchRayVolumes rv, float3 o, float3 d, float tSurface, float jitter)
{
    MarchResult result = NoMarch();
    if (rv.count == 0u)
    {
        return result;
    }
    float tStart;
    float tEnd;
    float step;
    MarchSpan(s, rv, tSurface, tStart, tEnd, step);
    bool ambient = (s.flags & kMarchNoIndirect) == 0u;
    float weightedDepth = 0.0f;
    float weight = 0.0f;
    uint i = 0u;
    for (uint n = 0u; n < kMarchMaxSteps; ++n)
    {
        if (!MarchStep(s, rv, o, d, tStart, tEnd, step, jitter, ambient, i, result, weightedDepth, weight))
        {
            break;
        }
    }
    result.depth = weight > 0.0f ? weightedDepth / weight : tStart;
    return result;
}

#endif
