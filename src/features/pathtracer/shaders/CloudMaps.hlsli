// The real-time clouds' maps (CloudMarcher.h, docs/architecture.md §8.19), as the integrator and the
// clouds' own passes read them. Shared by C++ and HLSL; the including code defines, before including
// this file:
//   float  CloudShadowSample(CloudData c, float2 uv)  CloudData::shadowMap, bilinear, wrapping
//   float4 CloudSkySample(CloudData c, float2 uv)     CloudData::skyMap, bilinear, clamping
//
// The shadow map holds, over one tile of the clouds (which repeat), the optical depth along the sun
// from a point of the cloud base (at its altitude: the base curves with the planet, CloudLookup.hlsli)
// up through the layer: below the base (all geometry is) the sun's transmittance is that of the point
// where the sun's ray crosses the base. The lookups need CloudLookup.hlsli included before. The sky map holds what the
// clouds add to (rgb, nits) and leave of (a) the sky behind them towards each direction of the upper
// hemisphere, seen from the camera: the clouds are kilometres away, so it serves every ray that leaves
// the scene upwards.
#ifndef AIR_PT_CLOUD_MAPS_HLSLI
#define AIR_PT_CLOUD_MAPS_HLSLI

#ifndef AIR_PT_CLOUD_MAPS_MAPPING_ONLY
#define AIR_PT_CLOUD_MAPS_MAPPING_ONLY 0
#endif

#include "air/renderer/shader/CloudShared.h"

// The upper hemisphere (y up) on the unit square: the octahedron's upper half, its diamond turned by
// 45 degrees to fill the square.
float2 CloudSkyMapUv(float3 d)
{
    float3 a = abs(d);
    float2 p = float2(d.x, d.z) / max(a.x + a.y + a.z, 1e-20f);
    return float2(p.x + p.y, p.x - p.y) * 0.5f + 0.5f;
}

float3 CloudSkyMapDirection(float2 uv)
{
    float2 a = uv * 2.0f - 1.0f;
    float2 p = float2(a.x + a.y, a.x - a.y) * 0.5f;
    float y = max(1.0f - abs(p.x) - abs(p.y), 0.0f);
    return normalize(float3(p.x, y, p.y));
}

#if !AIR_PT_CLOUD_MAPS_MAPPING_ONLY

// The sun's transmittance through the clouds from `p` (below their base) towards `wi`.
float CloudShadowMapTransmittance(CloudData c, float3 p, float3 wi)
{
    if (c.enabled == 0u || wi.y <= 0.0f || CloudAltitude(p) >= c.bottom)
    {
        return 1.0f;
    }
    float s = CloudCrossing(MakeCloudCurve(p, wi), c.bottom, 0.0f);
    if (s > kCloudMaxDistance)
    {
        return 1.0f;
    }
    float2 q = float2(p.x, p.z) + float2(wi.x, wi.z) * s;
    return exp(-max(CloudShadowSample(c, q / c.tileSize), 0.0f));
}

// What the clouds add to (rgb) and leave of (a) the sky towards `d`.
float4 CloudSkyMapLookup(CloudData c, float3 d)
{
    if (c.enabled == 0u || d.y <= 0.0f)
    {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return CloudSkySample(c, CloudSkyMapUv(d));
}

#endif

#endif
