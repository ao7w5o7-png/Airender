// Probe GI: traced reflections for smooth surfaces (ProbeGiShared.h; ReflectionTrace.hlsl, ReflectionResolve.hlsl).
// Surfaces smoother than kMaxRoughnessToTrace take their specular light from rays of their own; the probes' lobes
// (ProbeGather.hlsl) blur everything narrower than a probe cell. From kMaxRoughnessToTrace - kRoughnessFade on the
// two blend (ReflectionWeight).
#ifndef AIR_PROBE_GI_REFLECTION_COMMON_HLSLI
#define AIR_PROBE_GI_REFLECTION_COMMON_HLSLI

#include "GiCommon.hlsli"

static const float kMaxRoughnessToTrace = 0.4f; // perceptual (GGX alpha = roughness^2)
static const float kRoughnessFade = 0.1f;
static const float kMirrorRoughness = 0.05f; // below: the mirror direction, and no rays of neighbours reused

// The share of a pixel's specular light its traced rays give (the rest the probes').
float ReflectionWeight(float roughness)
{
    return saturate((kMaxRoughnessToTrace - roughness) / kRoughnessFade);
}

float ReflectionAlpha(float roughness)
{
    return max(roughness * roughness, 1e-4f);
}

// Visible normal sampling of GGX with spherical caps (Dupuy and Benyoub 2023). wo.z > 0.
float3 SampleGgxVndf(float3 wo, float alpha, float2 u)
{
    float3 woStd = normalize(float3(wo.x * alpha, wo.y * alpha, wo.z));
    float phi = 2.0f * kPi * u.x;
    float z = (1.0f - u.y) * (1.0f + woStd.z) - woStd.z;
    float sinTheta = sqrt(saturate(1.0f - z * z));
    float3 hStd = float3(sinTheta * cos(phi), sinTheta * sin(phi), z) + woStd;
    return normalize(float3(hStd.x * alpha, hStd.y * alpha, max(hStd.z, 1e-6f)));
}

float GgxD(float3 h, float alpha)
{
    float a2 = alpha * alpha;
    float d = h.z * h.z * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d);
}

float GgxLambda(float3 w, float alpha)
{
    float t2 = max(1.0f - w.z * w.z, 0.0f) / max(w.z * w.z, 1e-8f);
    return 0.5f * (sqrt(1.0f + alpha * alpha * t2) - 1.0f);
}

float GgxG1(float3 w, float alpha)
{
    return 1.0f / (1.0f + GgxLambda(w, alpha));
}

// The density of wi = reflect(-wo, h) for h drawn by SampleGgxVndf (solid angle measure, local frame).
float GgxVndfReflectionPdf(float3 wo, float3 wi, float alpha)
{
    float3 h = normalize(wo + wi);
    return GgxG1(wo, alpha) * GgxD(h, alpha) / (4.0f * max(wo.z, 1e-6f));
}

// The pixel a half resolution texel traces this frame (one of its 2 x 2, in turn).
uint2 ReflectionPixel(uint2 texel, uint frame)
{
    static const uint2 kOrder[4] = {uint2(0, 0), uint2(1, 1), uint2(1, 0), uint2(0, 1)};
    return texel * 2u + kOrder[(frame + texel.x * 3u + texel.y * 5u) & 3u];
}

#endif
