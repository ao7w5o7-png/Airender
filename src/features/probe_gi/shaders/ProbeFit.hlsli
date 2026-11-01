// Probe GI screen probes: how well a probe fits a pixel's surface (ProbeGiShared.h), for the gather
// (ProbeGather.hlsl) and for placing adaptive probes where the probes around fit nothing (ProbeAdapt.hlsl).
#ifndef AIR_PROBE_GI_PROBE_FIT_HLSLI
#define AIR_PROBE_GI_PROBE_FIT_HLSLI

#include "GiCommon.hlsli"

// The probe's distance from the pixel's plane (within a few per cent of their distance, a surface seen at an angle,
// and a few pixels' width; the plane of planeNormal, the surface's own where n is a normal map's) and how alike
// their normals are.
float FitWeight(float3 p, float3 n, float3 planeNormal, float footprint, ScreenProbe probe)
{
    if (probe.depth <= 0.0f)
    {
        return 0.0f;
    }
    float3 offset = probe.position - p;
    float plane = abs(dot(offset, planeNormal));
    float tolerance = 0.05f * length(offset) + 2.0f * footprint;
    float m = saturate(dot(n, UnpackNormal(probe.normal)));
    float m2 = m * m;
    return saturate(1.0f - plane / tolerance) * (m2 * m2 * m2 * m2);
}

// The normal of the surface the depth buffer holds at a pixel, towards the camera (direction `view`): from its
// neighbours' positions, on each axis the side nearer in depth; `fallback` where neither side is a surface.
float3 DepthNormal(GiFrame f, Texture2D<float> depth, uint2 pixel, float3 p, float z, float3 view, float3 fallback)
{
    float3 sides[2];
    for (uint axis = 0u; axis < 2u; ++axis)
    {
        int2 step = axis == 0u ? int2(1, 0) : int2(0, 1);
        float best = 1e30f;
        sides[axis] = float3(0.0f, 0.0f, 0.0f);
        for (int s = -1; s <= 1; s += 2)
        {
            int2 q = int2(pixel) + step * s;
            if (any(q < 0) || q.x >= int(f.width) || q.y >= int(f.height))
            {
                continue;
            }
            float zq = depth.Load(int3(q, 0));
            if (!(zq < 65000.0f) || zq <= 0.0f || abs(zq - z) >= best)
            {
                continue;
            }
            float3 direction;
            best = abs(zq - z);
            sides[axis] = (PixelPosition(f, uint2(q), zq, direction) - p) * float(s);
        }
    }
    float3 n = cross(sides[0], sides[1]);
    if (dot(n, n) <= 1e-20f)
    {
        return fallback;
    }
    n = normalize(n);
    return dot(n, view) > 0.0f ? -n : n;
}

// The bilinear weight of the 4 tiles' probes around a pixel times their fit, summed.
float TileFit(GiFrame f, StructuredBuffer<ScreenProbe> probes, uint2 pixel, float3 p, float3 n, float footprint)
{
    float2 tile = (float2(pixel) + 0.5f) / float(f.probeTile) - 0.5f;
    int2 base = int2(floor(tile));
    float2 w1 = tile - float2(base);
    float sum = 0.0f;
    for (uint k = 0u; k < 4u; ++k)
    {
        int2 corner = int2(k & 1u, k >> 1);
        int2 q = clamp(base + corner, int2(0, 0), int2(f.probesX - 1u, f.probesY - 1u));
        float2 wc = select(corner != 0, w1, 1.0f - w1);
        sum += wc.x * wc.y * FitWeight(p, n, n, footprint, probes[uint(q.y) * f.probesX + uint(q.x)]);
    }
    return sum;
}

// The adaptive probes of a tile: up to 4 indices (+ 1; 0: none) into the probes' buffer.
uint4 AdaptiveProbes(StructuredBuffer<uint4> headers, GiFrame f, int2 tile)
{
    if (any(tile < 0) || tile.x >= int(f.probesX) || tile.y >= int(f.probesY))
    {
        return uint4(0u, 0u, 0u, 0u);
    }
    return headers[uint(tile.y) * f.probesX + uint(tile.x)];
}

#endif
