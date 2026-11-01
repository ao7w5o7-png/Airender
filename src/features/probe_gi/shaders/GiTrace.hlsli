// Probe GI: a GI ray from a visible surface (ProbeGiShared.h). It marches the depth buffer first (what the
// camera sees is exact there, and its light is last frame's), then the voxels from where it lost sight of it,
// and past GiFrame::cacheDistance times the radiance cache's spacing it takes the cache's light. Rays that
// leave everything see the sky. The light is what the hit reflects towards the ray, never its emission.
// The including shader declares the push constants `g` (GiConstants) with:
//   input0 view depth (Texture2D<float>), input1 motion vectors (Texture2D<float2>),
//   input2 last frame's reflected light (Texture2D<float4>, nits; kInvalidIndex: none), input4 last frame's
//   surfaces (Texture2D<float4>: normal, view depth; kInvalidIndex: none).
#ifndef AIR_PROBE_GI_TRACE_HLSLI
#define AIR_PROBE_GI_TRACE_HLSLI

#include "VoxelTrace.hlsli"

static const float kSkyDepth = 65000.0f; // view depths from here on are the sky (the path tracer writes 65504)
static const float kScreenTraceFraction = 0.5f; // of the screen's width a screen trace may cross
// Voxels off the surface where the voxel trace starts, and over which its hit distances ramp up (door: a voxel off
// and the full distances from the start, -23.2%; half a voxel and the ramp, -20.9%; the other scenes alike).
static const float kGiOriginOffset = 0.5f;
static const float kGiHitRamp = 1.0f;
static const float kScreenEdgeFade = 0.05f;        // of the image's height: last frame's light fades out there
static const float kHistoryDepthTolerance = 0.02f; // relative: last frame's view depth at a screen hit

struct GiRay
{
    float3 radiance;
    float distance; // to the hit, or the far distance where it left
    uint source;    // 0 sky, 1 screen, 2 voxels, 3 cache
};

// Last frame's reflected light of the surface at `surface`, which the current frame shows at `texel`: found
// through the motion vectors, and only where last frame's pixel there showed that surface too (its view depth
// as expected: else what was there, an object that has moved or a disoccluded wall, would take its place) and
// lay inside the image (towards its edges, with a probability falling to 0 at the border, so that no seam shows
// where last frame's image ends).
bool PreviousLight(GiFrame f, uint2 texel, float3 surface, float u, out float3 radiance)
{
    radiance = float3(0.0f, 0.0f, 0.0f);
    float2 size = float2(f.width, f.height);
    Texture2D<float2> motion = ResourceDescriptorHeap[g.input1];
    float2 previous = float2(texel) + 0.5f + motion.Load(int3(texel, 0));
    float border = min(min(previous.x, size.x - previous.x), min(previous.y, size.y - previous.y));
    if (border < u * kScreenEdgeFade * size.y)
    {
        return false;
    }
    if (g.input4 != kInvalidIndex)
    {
        Texture2D<float4> surfacePast = ResourceDescriptorHeap[g.input4];
        float pastZ = surfacePast.Load(int3(min(uint2(previous), uint2(size) - 1u), 0)).w;
        float expected = CameraProject(f.previousCamera, surface, size).z;
        if (abs(pastZ - expected) > kHistoryDepthTolerance * expected + 2.0f * PixelFootprint(f, expected))
        {
            return false;
        }
    }
    Texture2D<float4> history = ResourceDescriptorHeap[g.input2];
    radiance = history.SampleLevel(gLinearClamp, previous / size, 0.0f).xyz / kHalfLightScale;
    return true;
}

// Marches the ray over the depth buffer (in screen space, depth interpolated as 1/z). Returns 1 with the
// hit's light, 2 when it went behind something the depth buffer cannot see past (tFree: how far it is
// known to be free), 0 when it left the screen or its reach (tFree likewise).
uint ScreenTrace(GiFrame f, float3 origin, float3 direction, uint2 startPixel, float jitter, out float3 radiance,
                 out float tFree)
{
    radiance = float3(0.0f, 0.0f, 0.0f);
    tFree = 0.0f;
    float2 size = float2(f.width, f.height);
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    float3 p0 = CameraProject(f.camera, origin, size);
    if (p0.z <= 0.0f)
    {
        return 0u;
    }
    // The far end: clipped to the near plane, then the screen segment to the screen's reach.
    float tMax = 1000.0f;
    float near = 0.01f;
    float dz = dot(direction, f.camera.forward.xyz);
    if (dz < 0.0f)
    {
        tMax = min(tMax, (p0.z - near) / -dz);
    }
    float3 p1 = CameraProject(f.camera, origin + direction * tMax, size);
    float2 delta = p1.xy - p0.xy;
    float length2 = length(delta);
    float reach = kScreenTraceFraction * size.x;
    float s1 = 1.0f;
    if (length2 > reach)
    {
        s1 = reach / length2;
    }
    // Clip to the screen.
    float2 dInv = 1.0f / select(abs(delta) > 1e-6f, delta, 1e-6f);
    float2 ta = (float2(0.0f, 0.0f) - p0.xy) * dInv;
    float2 tb = (size - p0.xy) * dInv;
    float2 tExit = max(ta, tb);
    s1 = min(s1, min(tExit.x, tExit.y));
    if (s1 <= 0.0f)
    {
        return 0u;
    }
    float invZ0 = 1.0f / p0.z;
    float invZ1 = 1.0f / p1.z;
    uint steps = max(f.screenSteps, 1u);
    float lastFree = 0.0f;
    for (uint i = 1u; i <= steps; ++i)
    {
        // Steps grow towards the end: fine near the origin, where contact shadows are.
        float u = (float(i) - 1.0f + jitter) / float(steps);
        float s = s1 * u * u;
        float2 xy = p0.xy + delta * s;
        uint2 texel = uint2(clamp(xy, 0.0f, size - 1.0f));
        if (all(texel == startPixel))
        {
            continue;
        }
        float rayZ = 1.0f / lerp(invZ0, invZ1, s);
        float sceneZ = depth.Load(int3(texel, 0));
        // The parameter along the ray in metres (perspective: from the view depth along the ray; a ray
        // parallel to the image plane keeps its depth and is linear on screen).
        float t = abs(dz) > 1e-4f ? (rayZ - p0.z) / dz : s * tMax;
        if (rayZ > sceneZ && sceneZ < kSkyDepth)
        {
            float thickness = f.screenThickness * sceneZ;
            if (rayZ - sceneZ < thickness)
            {
                if (g.input2 == kInvalidIndex)
                {
                    tFree = lastFree;
                    return 2u;
                }
                float3 view;
                float3 surface = PixelPosition(f, texel, sceneZ, view);
                if (!PreviousLight(f, texel, surface, jitter, radiance))
                {
                    tFree = lastFree;
                    return 2u;
                }
                tFree = max(t, 0.0f);
                return 1u;
            }
            tFree = lastFree;
            return 2u; // behind an object: unknown
        }
        lastFree = max(t, 0.0f);
    }
    tFree = lastFree;
    return 0u;
}

// The first part of a GI ray from a visible surface point with normal `normal`: the depth buffer. True with the
// ray's light when it hit there; else tStart is where the voxels take over.
bool TraceGiScreen(GiFrame f, float3 position, float3 normal, float3 direction, uint2 pixel, float footprint,
                   float jitter, out GiRay result, out float tStart)
{
    result.radiance = float3(0.0f, 0.0f, 0.0f);
    result.distance = 0.0f;
    result.source = 0u;
    tStart = 0.0f;
    if ((f.flags & kGiScreenTraces) == 0u)
    {
        return false;
    }
    float3 radiance;
    float tFree;
    uint screen = ScreenTrace(f, position + normal * footprint, direction, pixel, jitter, radiance, tFree);
    if (screen == 1u)
    {
        result.radiance = radiance;
        result.distance = tFree;
        result.source = 1u;
        return true;
    }
    tStart = tFree;
    return false;
}

// The rest of it: the voxels from tStart on, from an origin half a voxel off the surface; the hit distances ramp up
// over the first voxel (else the surface's own band is a hit).
GiRay TraceGiVoxels(GiFrame f, float3 position, float3 normal, float3 direction, float tStart)
{
    GiRay result;
    uint first = FindCascade(f, position, kCascadeMargin);
    float voxel = first < f.cascadeCount ? f.cascades[first].voxelSize : 1.0f;
    float3 origin = position + normal * (kGiOriginOffset * voxel);
    gHitRampVoxels = kGiHitRamp;
    float tCache = 1e30f;
    if ((f.flags & kGiCache) != 0u)
    {
        uint level = FindCacheLevel(f, origin);
        if (level < f.cascadeCount)
        {
            tCache = max(f.cacheDistance * CacheSpacing(f.cascades[level]), tStart);
        }
    }
    result.radiance = TraceLight(f, origin, direction, tStart, tCache, 0u, result.distance, result.source);
    gHitRampVoxels = 0.0f;
    return result;
}
#endif
