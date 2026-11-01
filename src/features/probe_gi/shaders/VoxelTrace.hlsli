// Probe GI: rays through the voxel scene's distance field, the light of the voxels they hit and of the
// radiance cache (ProbeGiShared.h). HLSL only.
//
// A ray starts in the finest cascade that holds its origin and moves to the next coarser cascade where it
// leaves one. Within a cascade it leaps over empty bricks (their Chebyshev distance to the nearest brick
// with a surface) and sphere traces the distance field (trilinear, exact up to kVoxelBand voxels) in the
// others, until the distance falls below kHitDistance voxels.
#ifndef AIR_PROBE_GI_VOXEL_TRACE_HLSLI
#define AIR_PROBE_GI_VOXEL_TRACE_HLSLI

#include "GiCommon.hlsli"

// Voxels: closer to a triangle than this the surface voxels' planes decide (SurfacePlaneDistance). The field
// is unsigned and filtered: where a plane lies halfway between two voxel centres (on a voxel face), the
// interpolated distance never drops below 0.5.
static const float kHitDistance = 0.65f;
// Below this the field is a hit without asking the planes (a gap narrower than about twice this closes).
static const float kCertainHitDistance = 0.35f;
static const float kPlaneExtent = 0.5f;      // voxels from its voxel's centre a face's plane reaches (its cube)
static const float kPlaneHitDistance = 0.1f; // voxels from a surface voxel's plane: a hit (thinner keeps gaps open)
static const float kPlaneMinStep = 0.2f;     // voxels
static const float kMinStep = 0.25f;      // voxels
// Steps per cascade; a ray out of them carries on in the next coarser one. Rays grazing a plane that runs through
// every cascade (sunset's ground) take most: 96 cost 40% more there for no measurable change anywhere.
static const uint kMaxStepsPerCascade = 48u;
static const float kCascadeMargin = 1.0f; // voxels at a cascade's faces its traces leave alone (toroidal seams)
// Rays that leave a surface: their hit distances grow from 0 at the origin to the full ones a voxel (of the
// cascade they are in) out, so that the surface they start from is not a hit although the origin lies within its
// band. Set by the passes whose rays start on a surface; 0: the full distances from the start.
static float gHitRampVoxels = 0.0f;

static const float3 kFaceAxes[6] = {float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0),
                                    float3(0, -1, 0), float3(0, 0, 1),  float3(0, 0, -1)};

int3 CascadeOrigin(CascadeData c)
{
    return int3(c.originX, c.originY, c.originZ);
}

// Storage coordinates of world voxel v (toroidal).
uint3 VoxelStorage(int3 v)
{
    return uint3(v & int(kVoxelResolution - 1u));
}

uint3 BrickStorage(int3 b)
{
    return uint3(b & int(kBrickResolution - 1u));
}

// The cascade's box in metres, shrunk by kCascadeMargin voxels.
void CascadeBox(CascadeData c, out float3 lo, out float3 hi)
{
    float3 origin = float3(CascadeOrigin(c));
    lo = (origin + kCascadeMargin) * c.voxelSize;
    hi = (origin + float(kVoxelResolution) - kCascadeMargin) * c.voxelSize;
}

bool InCascade(CascadeData c, float3 p, float marginVoxels)
{
    float3 v = p / c.voxelSize - float3(CascadeOrigin(c));
    return all(v >= marginVoxels) && all(v <= float(kVoxelResolution) - marginVoxels);
}

// The finest built cascade holding p at least marginVoxels inside; cascadeCount when none does.
uint FindCascade(GiFrame f, float3 p, float marginVoxels)
{
    for (uint i = 0u; i < f.cascadeCount; ++i)
    {
        if (f.cascades[i].valid != 0u && InCascade(f.cascades[i], p, marginVoxels))
        {
            return i;
        }
    }
    return f.cascadeCount;
}

// Entry and exit of a ray (inverse direction rd) with a box.
float2 RayBox(float3 o, float3 rd, float3 lo, float3 hi)
{
    float3 t0 = (lo - o) * rd;
    float3 t1 = (hi - o) * rd;
    float3 tn = min(t0, t1);
    float3 tf = max(t0, t1);
    return float2(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}

float SampleDistance(CascadeData c, float3 v)
{
    Texture3D<float> sdf = ResourceDescriptorHeap[c.sdf];
    return sdf.SampleLevel(gLinearWrap, v / float(kVoxelResolution), 0.0f) * kVoxelBand;
}

// A surface voxel, also while the running pass writes them (gVoxelsWritable, defined below).
SurfaceVoxel LoadSurfaceVoxel(CascadeData c, uint cascade, uint slot);

// The distance (voxels) from v to the nearest surface the faces of the surface voxel v lies in show, each the
// plane of its sides within the voxel (its normal and offset, ProbeGiShared.h); 1e30 when v's voxel holds no
// surface (one nearer in a neighbour shows in that neighbour's planes at the next step). Unlike the filtered
// field this is exact for a plane on a voxel's face and leaves a gap a voxel wide open (a door ajar between two
// rooms: the voxel in the gap holds both walls' planes).
float SurfacePlaneDistance(GiFrame f, uint cascade, float3 v)
{
    CascadeData c = f.cascades[cascade];
    Texture3D<uint> index = ResourceDescriptorHeap[c.voxelIndex];
    int3 u = int3(floor(v));
    uint slot = index.Load(int4(uint3(u & int(kVoxelResolution - 1u)), 0));
    if (slot == 0u)
    {
        return 1e30f;
    }
    SurfaceVoxel s = LoadSurfaceVoxel(c, cascade, slot - 1u);
    uint mask = s.coord >> 24;
    float3 centre = float3(u) + 0.5f;
    float best = 1e30f;
    for (uint face = 0u; face < 6u; ++face)
    {
        if ((mask & (1u << face)) == 0u)
        {
            continue;
        }
        float offset;
        float3 n = UnpackNormalOffset(s.normal[face], offset);
        float signedDistance = dot(v - centre, n) - offset;
        float3 onPlane = v - n * signedDistance;
        if (all(abs(onPlane - centre) <= kPlaneExtent))
        {
            best = min(best, abs(signedDistance));
        }
    }
    return best;
}

struct VoxelHit
{
    bool hit;
    float t;        // metres along the ray
    uint cascade;   // where it hit
    float3 voxel;   // the hit point in the cascade's voxel coordinates
    uint steps;     // samples taken (debugging)
};

// The first surface of the voxel scene along origin + t direction, t in [tMin, tMax], starting in the
// finest cascade that holds the point at tMin, or in firstCascade when that is coarser (the surface voxels
// of a coarse cascade light themselves through it: cheaper, and as coarse as the light they hold).
VoxelHit TraceVoxelsFrom(GiFrame f, float3 origin, float3 direction, float tMin, float tMax, uint firstCascade)
{
    VoxelHit result;
    result.hit = false;
    result.t = tMax;
    result.cascade = 0u;
    result.voxel = float3(0.0f, 0.0f, 0.0f);
    result.steps = 0u;
    float3 rd = 1.0f / select(abs(direction) > 1e-8f, direction, 1e-8f);
    float t = tMin;
    float rampScale = gHitRampVoxels > 0.0f ? 1.0f / gHitRampVoxels : 0.0f;
    float rampBias = gHitRampVoxels > 0.0f ? 0.0f : 1.0f;
    uint first = max(FindCascade(f, origin + direction * t, kCascadeMargin), firstCascade);
    for (uint ci = first; ci < f.cascadeCount && t < tMax; ++ci)
    {
        CascadeData c = f.cascades[ci];
        if (c.valid == 0u)
        {
            continue;
        }
        float3 lo;
        float3 hi;
        CascadeBox(c, lo, hi);
        float2 span = RayBox(origin, rd, lo, hi);
        if (span.y <= max(span.x, t))
        {
            continue;
        }
        t = max(t, span.x);
        float tEnd = min(span.y, tMax);
        Texture3D<uint> bricks = ResourceDescriptorHeap[c.bricks];
        float inv = 1.0f / c.voxelSize;
        float3 ov = origin * inv; // the ray in voxels: t stays in metres
        float3 dv = direction;    // voxels per metre is inv, applied to t below
        for (uint s = 0u; s < kMaxStepsPerCascade && t < tEnd; ++s)
        {
            ++result.steps;
            float3 v = ov + dv * (t * inv);
            int3 brick = int3(floor(v / float(kVoxelBrick)));
            uint distance = bricks.Load(int4(BrickStorage(brick), 0));
            if (distance > 0u)
            {
                // No surface within distance - 1 bricks: leave the cube of empty bricks around this one.
                float3 cubeLo = float3(brick - int(distance - 1u)) * float(kVoxelBrick);
                float3 cubeHi = float3(brick + int(distance)) * float(kVoxelBrick);
                float2 cube = RayBox(ov, rd, cubeLo, cubeHi);
                t = max(cube.y * c.voxelSize, t + kMinStep * c.voxelSize) + 1e-4f * c.voxelSize;
                continue;
            }
            float d = SampleDistance(c, v);
            // Near the origin the hit distances ramp up (gHitRampVoxels).
            float ramp = saturate(t * inv * rampScale + rampBias);
            if (d < kCertainHitDistance * ramp)
            {
                result.hit = true;
                result.t = t;
                result.cascade = ci;
                result.voxel = v;
                return result;
            }
            if (d < kHitDistance * ramp)
            {
                // Between: a plane on a voxel's face, or a gap; the surface voxels' planes say which.
                float plane = SurfacePlaneDistance(f, ci, v);
                bool hit = plane < 1e29f ? plane < kPlaneHitDistance : d < 0.5f * ramp;
                if (hit)
                {
                    result.hit = true;
                    result.t = t;
                    result.cascade = ci;
                    result.voxel = v;
                    return result;
                }
                // Steps no longer than the distance to the planes, at least kPlaneMinStep: from outside a
                // plane's band (+-kPlaneHitDistance) such a step lands at most kPlaneMinStep - kPlaneHitDistance
                // beyond the plane, inside the band.
                t += (plane < 1e29f ? clamp(plane * 0.9f, kPlaneMinStep, 0.5f) : max(d * 0.8f, kMinStep)) * c.voxelSize;
                continue;
            }
            t += max(d * 0.9f, kMinStep) * c.voxelSize;
        }
        if (t < tEnd)
        {
            // Out of steps in this cascade: carry on in the next one from here.
        }
    }
    result.t = min(t, tMax);
    return result;
}

VoxelHit TraceVoxels(GiFrame f, float3 origin, float3 direction, float tMin, float tMax)
{
    return TraceVoxelsFrom(f, origin, direction, tMin, tMax, 0u);
}

// Whether a shadow ray reaches tMax (true: nothing in the way).
bool VoxelsVisible(GiFrame f, float3 origin, float3 direction, float tMin, float tMax)
{
    return !TraceVoxels(f, origin, direction, tMin, tMax).hit;
}

bool VoxelsVisibleFrom(GiFrame f, float3 origin, float3 direction, float tMin, float tMax, uint firstCascade)
{
    return !TraceVoxelsFrom(f, origin, direction, tMin, tMax, firstCascade).hit;
}

// --- Surface voxels -------------------------------------------------------------------------------

// Set by the pass that writes every cascade's surface voxels (VoxelLight.hlsl): they are read through their
// UAVs (CascadeData::voxelsWritable) rather than their SRVs.
static bool gVoxelsWritable = false;

SurfaceVoxel LoadSurfaceVoxel(CascadeData c, uint cascade, uint slot)
{
    if (gVoxelsWritable)
    {
        RWStructuredBuffer<SurfaceVoxel> writable = ResourceDescriptorHeap[c.voxelsWritable];
        return writable[slot];
    }
    StructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[c.voxels];
    return voxels[slot];
}

// The light the voxels around a hit point of cascade `cascade` reflect back along the ray (towards
// -direction): the 2 x 2 x 2 voxels around the point, trilinearly, each through its faces that face the ray.
// `found` is false when none of them has a face towards the ray (backface: the ray started inside a mesh) or
// none is a surface.
float3 VoxelRadiance(GiFrame f, uint cascade, float3 v, float3 direction, out bool found, out bool backface)
{
    CascadeData c = f.cascades[cascade];
    Texture3D<uint> index = ResourceDescriptorHeap[c.voxelIndex];
    float3 base = floor(v - 0.5f);
    float3 w1 = v - 0.5f - base;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    float away = 0.0f;
    for (uint k = 0u; k < 8u; ++k)
    {
        int3 corner = int3(k & 1u, (k >> 1) & 1u, k >> 2);
        float3 wc = select(corner != 0, w1, 1.0f - w1);
        float wTri = wc.x * wc.y * wc.z;
        uint slot = index.Load(int4(VoxelStorage(int3(base) + corner), 0));
        if (slot == 0u)
        {
            continue;
        }
        SurfaceVoxel s = LoadSurfaceVoxel(c, cascade, slot - 1u);
        uint mask = s.coord >> 24;
        for (uint face = 0u; face < 6u; ++face)
        {
            if ((mask & (1u << face)) == 0u)
            {
                continue;
            }
            float facing = -dot(direction, kFaceAxes[face]);
            float coverage = float(s.albedo[face] >> 24) / 255.0f;
            if (facing > 0.0f)
            {
                float w = wTri * facing * coverage;
                sum += UnpackRadiance(s.radiance[face]) * w;
                weight += w;
            }
            else
            {
                away += wTri * -facing * coverage;
            }
        }
    }
    found = weight > 0.0f;
    backface = !found && away > 0.0f;
    return found ? sum / weight : float3(0.0f, 0.0f, 0.0f);
}

// The same for the albedo the faces show (the debug views).
float3 VoxelAlbedo(CascadeData c, float3 v, float3 direction, out bool found)
{
    Texture3D<uint> index = ResourceDescriptorHeap[c.voxelIndex];
    StructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[c.voxels];
    float3 base = floor(v - 0.5f);
    float3 w1 = v - 0.5f - base;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    for (uint k = 0u; k < 8u; ++k)
    {
        int3 corner = int3(k & 1u, (k >> 1) & 1u, k >> 2);
        float3 wc = select(corner != 0, w1, 1.0f - w1);
        uint slot = index.Load(int4(VoxelStorage(int3(base) + corner), 0));
        if (slot == 0u)
        {
            continue;
        }
        SurfaceVoxel s = voxels[slot - 1u];
        uint mask = s.coord >> 24;
        for (uint face = 0u; face < 6u; ++face)
        {
            float facing = -dot(direction, kFaceAxes[face]);
            if ((mask & (1u << face)) != 0u && facing > 0.0f)
            {
                float4 a = UnpackUnorm4(s.albedo[face]);
                float w = wc.x * wc.y * wc.z * facing * a.w;
                sum += a.xyz * w;
                weight += w;
            }
        }
    }
    found = weight > 0.0f;
    return found ? sum / weight : float3(0.0f, 0.0f, 0.0f);
}

// --- Radiance cache ---------------------------------------------------------------------------------

// Octahedral texel coordinates (0..1) of a direction over the whole sphere.
float2 CacheUv(float3 direction)
{
    return OctEncode(direction) * 0.5f + 0.5f;
}

float3 CacheDirection(float2 uv)
{
    return OctDecode(uv * 2.0f - 1.0f);
}

int3 CacheOrigin(CascadeData c)
{
    return CascadeOrigin(c) / 8;
}

float CacheSpacing(CascadeData c)
{
    return c.voxelSize * 8.0f;
}

// Probe (world probe coordinates) -> its position.
float3 CacheProbePosition(CascadeData c, int3 probe)
{
    return (float3(probe) + 0.5f) * CacheSpacing(c);
}

uint CacheProbeIndex(int3 probe)
{
    uint3 s = uint3(probe & int(kCacheResolution - 1u));
    return (s.z * kCacheResolution + s.y) * kCacheResolution + s.x;
}

// Atlas texel of a probe's octahedral texel (probe storage index -> a tile of kCacheTexels^2).
uint2 CacheAtlasTexel(uint probeIndex, uint2 texel)
{
    uint tilesX = kCacheResolution * kCacheResolution;
    uint2 tile = uint2(probeIndex % tilesX, probeIndex / tilesX);
    return tile * kCacheTexels + texel;
}

// Bilinear radiance of one probe towards `direction`, texels clamped within its tile.
float4 CacheProbeRadiance(CascadeData c, uint probeIndex, float3 direction)
{
    Texture2D<float4> atlas = ResourceDescriptorHeap[c.cacheRadiance];
    float2 uv = CacheUv(direction) * float(kCacheTexels) - 0.5f;
    float2 base = floor(uv);
    float2 w = uv - base;
    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint k = 0u; k < 4u; ++k)
    {
        int2 corner = int2(k & 1u, k >> 1);
        int2 texel = clamp(int2(base) + corner, 0, int(kCacheTexels) - 1);
        float2 wc = select(corner != 0, w, 1.0f - w);
        sum += atlas.Load(int3(CacheAtlasTexel(probeIndex, uint2(texel)), 0)) * (wc.x * wc.y);
    }
    return float4(sum.xyz / kHalfLightScale, sum.w);
}

// The finest cache level holding p a probe inside its lattice; cascadeCount when none does.
uint FindCacheLevel(GiFrame f, float3 p)
{
    for (uint i = 0u; i < f.cascadeCount; ++i)
    {
        CascadeData c = f.cascades[i];
        float3 q = p / CacheSpacing(c) - 0.5f - float3(CacheOrigin(c));
        if (c.valid != 0u && all(q >= 0.0f) && all(q <= float(kCacheResolution) - 1.0f))
        {
            return i;
        }
    }
    return f.cascadeCount;
}

float CacheVisibility(CascadeData c, uint probeIndex, float3 direction, float distance);

// The mean distance a probe's rays went towards `direction` (its texel's).
float CacheProbeDistance(CascadeData c, uint probeIndex, float3 direction)
{
    Texture2D<float2> moments = ResourceDescriptorHeap[c.cacheDistance];
    uint2 texel = min(uint2(CacheUv(direction) * float(kCacheTexels)), kCacheTexels - 1u);
    return moments.Load(int3(CacheAtlasTexel(probeIndex, texel), 0)).x;
}

// The light arriving at p from `direction` as the cache's probes around p saw it (trilinear over the
// valid ones that see p). False when p is outside every level or no probe around it is valid. Where the
// probes' rays that way left the level (they saw the sky), the sky itself: their 8 x 8 texels blur it.
bool CacheRadiance(GiFrame f, float3 p, float3 direction, out float3 radiance)
{
    radiance = float3(0.0f, 0.0f, 0.0f);
    uint level = FindCacheLevel(f, p);
    if (level >= f.cascadeCount)
    {
        return false;
    }
    CascadeData c = f.cascades[level];
    float3 q = p / CacheSpacing(c) - 0.5f;
    float3 base = floor(q);
    float3 w1 = q - base;
    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float far = 0.0f;
    for (uint k = 0u; k < 8u; ++k)
    {
        int3 corner = int3(k & 1u, (k >> 1) & 1u, k >> 2);
        float3 wc = select(corner != 0, w1, 1.0f - w1);
        int3 probe = int3(base) + corner;
        uint index = CacheProbeIndex(probe);
        float4 r = CacheProbeRadiance(c, index, direction);
        // The probe's validity (a), and whether it sees p (a probe behind a wall sees another room).
        float3 toPoint = p - CacheProbePosition(c, probe);
        float distance = length(toPoint);
        float visible = distance > 1e-5f ? CacheVisibility(c, index, toPoint / distance, distance) : 1.0f;
        float w = wc.x * wc.y * wc.z * r.w * (visible + 1e-3f);
        sum += float4(r.xyz, 1.0f) * w;
        far += CacheProbeDistance(c, index, direction) * w;
    }
    if (sum.w <= 1e-4f)
    {
        return false;
    }
    radiance = far / sum.w >= 0.9f * CacheSpacing(c) * float(kCacheResolution) ? SkyLight(f, direction) : sum.xyz / sum.w;
    return true;
}

// The light arriving along a ray: traced through the voxels (from firstCascade or a finer one, GiTrace's
// rules) up to tCache, then the radiance cache's where the ray is still free and probes there see it, else on
// through the voxels; the sky's when it leaves them all. Never what the hit emits.
float3 TraceLight(GiFrame f, float3 origin, float3 direction, float tStart, float tCache, uint firstCascade,
                  out float distance, out uint source)
{
    VoxelHit hit = TraceVoxelsFrom(f, origin, direction, tStart, tCache, firstCascade);
    bool found;
    bool backface;
    if (hit.hit)
    {
        distance = hit.t;
        source = 2u;
        return VoxelRadiance(f, hit.cascade, hit.voxel, direction, found, backface);
    }
    if (tCache < 1e30f && hit.t >= tCache * 0.999f)
    {
        float3 radiance;
        if (CacheRadiance(f, origin + direction * hit.t, direction, radiance))
        {
            distance = hit.t;
            source = 3u;
            return radiance;
        }
        hit = TraceVoxelsFrom(f, origin, direction, hit.t, 1e30f, firstCascade);
        if (hit.hit)
        {
            distance = hit.t;
            source = 2u;
            return VoxelRadiance(f, hit.cascade, hit.voxel, direction, found, backface);
        }
    }
    distance = 1e30f;
    source = 0u;
    return SkyLight(f, direction);
}

// --- Spherical harmonics ----------------------------------------------------------------------------

void ShBasis(float3 d, out float b[9])
{
    b[0] = 0.282095f;
    b[1] = 0.488603f * d.y;
    b[2] = 0.488603f * d.z;
    b[3] = 0.488603f * d.x;
    b[4] = 1.092548f * d.x * d.y;
    b[5] = 1.092548f * d.y * d.z;
    b[6] = 0.315392f * (3.0f * d.z * d.z - 1.0f);
    b[7] = 1.092548f * d.x * d.z;
    b[8] = 0.546274f * (d.x * d.x - d.y * d.y);
}

// Evaluates SH coefficients (rgb each) stored as ProbeSh for direction n.
float3 ShEvaluate(ProbeSh sh, float3 n)
{
    float b[9];
    ShBasis(n, b);
    float3 c[9];
    c[0] = sh.c0.xyz;
    c[1] = float3(sh.c0.w, sh.c1.x, sh.c1.y);
    c[2] = float3(sh.c1.z, sh.c1.w, sh.c2.x);
    c[3] = float3(sh.c2.y, sh.c2.z, sh.c2.w);
    c[4] = sh.c3.xyz;
    c[5] = float3(sh.c3.w, sh.c4.x, sh.c4.y);
    c[6] = float3(sh.c4.z, sh.c4.w, sh.c5.x);
    c[7] = float3(sh.c5.y, sh.c5.z, sh.c5.w);
    c[8] = sh.c6;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (uint i = 0u; i < 9u; ++i)
    {
        sum += c[i] * b[i];
    }
    return sum;
}

ProbeSh ShPack(float3 c[9], float valid)
{
    ProbeSh sh;
    sh.c0 = float4(c[0], c[1].x);
    sh.c1 = float4(c[1].y, c[1].z, c[2].x, c[2].y);
    sh.c2 = float4(c[2].z, c[3]);
    sh.c3 = float4(c[4], c[5].x);
    sh.c4 = float4(c[5].y, c[5].z, c[6].x, c[6].y);
    sh.c5 = float4(c[6].z, c[7]);
    sh.c6 = c[8];
    sh.valid = valid;
    sh.normalIrradiance = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return sh;
}

// Chebyshev's bound on the visibility between a cache probe and a point `distance` away towards
// `direction`, from the mean and squared distance of the probe's rays that way (Majercik et al. 2019).
float CacheVisibility(CascadeData c, uint probeIndex, float3 direction, float distance)
{
    Texture2D<float2> moments = ResourceDescriptorHeap[c.cacheDistance];
    float2 uv = CacheUv(direction) * float(kCacheTexels);
    uint2 texel = min(uint2(uv), kCacheTexels - 1u);
    float2 m = moments.Load(int3(CacheAtlasTexel(probeIndex, texel), 0));
    if (distance <= m.x)
    {
        return 1.0f;
    }
    float variance = abs(m.y - m.x * m.x);
    float delta = distance - m.x;
    float v = variance / (variance + delta * delta);
    return max(v * v * v, 0.0f);
}

// The irradiance at p for normal n from the cache's probes around p: trilinear, weighed towards the probes
// in front of the surface and by their visibility. False when none is valid there.
bool CacheIrradiance(GiFrame f, float3 p, float3 n, out float3 irradiance)
{
    irradiance = float3(0.0f, 0.0f, 0.0f);
    uint level = FindCacheLevel(f, p);
    if (level >= f.cascadeCount)
    {
        return false;
    }
    CascadeData c = f.cascades[level];
    StructuredBuffer<ProbeSh> probes = ResourceDescriptorHeap[c.cacheIrradiance];
    float spacing = CacheSpacing(c);
    float3 q = p / spacing - 0.5f;
    float3 base = floor(q);
    float3 w1 = q - base;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    for (uint k = 0u; k < 8u; ++k)
    {
        int3 corner = int3(k & 1u, (k >> 1) & 1u, k >> 2);
        int3 probe = int3(base) + corner;
        uint index = CacheProbeIndex(probe);
        ProbeSh sh = probes[index];
        if (sh.valid <= 0.0f)
        {
            continue;
        }
        float3 wc = select(corner != 0, w1, 1.0f - w1);
        float3 toProbe = CacheProbePosition(c, probe) - p;
        float distance = length(toProbe);
        float3 dir = distance > 1e-5f ? toProbe / distance : n;
        // Probes behind the surface see the other side of it.
        float facing = (dot(dir, n) + 1.0f) * 0.5f;
        float w = wc.x * wc.y * wc.z * (facing * facing + 0.05f);
        w *= CacheVisibility(c, index, -dir, distance) + 1e-3f;
        sum += max(ShEvaluate(sh, n), 0.0f) * w;
        weight += w;
    }
    if (weight <= 1e-6f)
    {
        return false;
    }
    irradiance = sum / weight;
    return true;
}

#endif
