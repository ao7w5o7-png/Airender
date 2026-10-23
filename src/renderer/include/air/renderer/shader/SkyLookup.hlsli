// Sky radiance and the sun's disc from SkyData and its table (SkyShared.h). Shared by C++ and HLSL;
// the including code defines, before including this file:
//   float3 SkyTableLoad(uint texel)   linear sRGB radiance of texel y * width + x, nits
#ifndef AIR_RENDERER_SKY_LOOKUP_HLSLI
#define AIR_RENDERER_SKY_LOOKUP_HLSLI

#include "air/renderer/shader/SkyShared.h"

// Radiance of the sky (without the sun's disc) towards `direction`, bilinear between texel centres.
float3 SkyRadiance(SkyData sky, float3 direction)
{
    if (sky.width == 0u)
    {
        return sky.uniformRadiance;
    }
    float2 uv = SkyTableUv(direction, sky.sunDirection);
    float x = clamp(uv.x * float(sky.width) - 0.5f, 0.0f, float(sky.width - 1u));
    float y = clamp(uv.y * float(sky.height) - 0.5f, 0.0f, float(sky.height - 1u));
    uint x0 = min(uint(x), sky.width - 2u);
    uint y0 = min(uint(y), sky.height - 2u);
    float fx = x - float(x0);
    float fy = y - float(y0);
    uint row0 = y0 * sky.width;
    uint row1 = row0 + sky.width;
    float3 bottom = lerp(SkyTableLoad(row0 + x0), SkyTableLoad(row0 + x0 + 1u), fx);
    float3 top = lerp(SkyTableLoad(row1 + x0), SkyTableLoad(row1 + x0 + 1u), fx);
    return lerp(bottom, top, fy);
}

bool HasSun(SkyData sky)
{
    return sky.sunRadiance.x + sky.sunRadiance.y + sky.sunRadiance.z > 0.0f;
}

// Solid angle of the sun's disc: the inverse pdf of sampling it uniformly.
float SunSolidAngle(SkyData sky)
{
    return 2.0f * kSkyPi * (1.0f - sky.sunCosRadius);
}

// A direction uniformly within the sun's disc (cone sampling).
float3 SampleSunDirection(SkyData sky, float2 u)
{
    float cosTheta = 1.0f - u.x * (1.0f - sky.sunCosRadius);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * kSkyPi * u.y;
    float3 w = sky.sunDirection;
    // Orthonormal basis around w (Duff et al. 2017).
    float s = w.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + w.z);
    float b = w.x * w.y * a;
    float3 t = float3(1.0f + s * w.x * w.x * a, s * b, -s * w.x);
    float3 bt = float3(b, s + w.y * w.y * a, -w.y);
    return t * (sinTheta * cos(phi)) + bt * (sinTheta * sin(phi)) + w * cosTheta;
}

#endif
