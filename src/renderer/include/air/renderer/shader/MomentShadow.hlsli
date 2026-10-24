// Moment shadow maps of volumes (MomentShadowShared.h), shared by C++ and HLSL: where a point lies in
// a map, a texel's ray, the moments a march along it adds up, and the transmittance a lookup
// reconstructs from them. The including code defines, before including this file:
//   float  MomentMapB0Sample(MomentShadowMap m, float2 uv)  slice m.slice of the b0 array, bilinear
//   float4 MomentMapBSample(MomentShadowMap m, float2 uv)   the b1..b4 array likewise
//
// The moments of a ray: b0 = sum of the optical depths dTau of its steps, b_k = sum of dTau z^k for
// k = 1..4, z in [-1, 1] the step's depth along the ray. They add up in any order and mix linearly,
// so maps of raw moments filter correctly with the hardware's bilinear filter (the moments of a mix
// of rays are the mix of their moments). From b0 and the normalized moments b_k / b0 the share of the
// optical depth in front of a depth z is bounded by the three-point distributions with these moments
// that have a point at z (Markov-Krein; Peters and Klein 2015): MomentAbsorbanceShare counts the points
// in front of z and a share `overestimation` of the one at z, as moment-based order-independent
// transparency does (Muenstermann et al. 2018), with its moments pulled slightly towards a safe
// interior vector first so that rounding cannot leave the set of valid moments.
#ifndef AIR_RENDERER_MOMENT_SHADOW_HLSLI
#define AIR_RENDERER_MOMENT_SHADOW_HLSLI

#include "air/renderer/shader/MomentShadowShared.h"

// The moments are stored as half floats: rounded to 11 significant bits they need a stronger pull
// than the 5e-7 of single precision (MomentTests.cpp measures it).
AIR_CONST float kMomentBias = 1e-3f;
AIR_CONST float kMomentOverestimation = 0.25f; // the point at z counts this much (0: a lower bound)
AIR_CONST float kMomentEmpty = 1e-4f;          // b0 below this: no volume along the texel's ray

// The octahedral map of the unit sphere onto [0, 1]^2 (Cigolle et al. 2014) and back.
float2 OctahedralUv(float3 d)
{
    float s = abs(d.x) + abs(d.y) + abs(d.z);
    float px = d.x / s;
    float py = d.y / s;
    if (d.z < 0.0f)
    {
        float fx = (1.0f - abs(py)) * (px >= 0.0f ? 1.0f : -1.0f);
        float fy = (1.0f - abs(px)) * (py >= 0.0f ? 1.0f : -1.0f);
        px = fx;
        py = fy;
    }
    return float2(px * 0.5f + 0.5f, py * 0.5f + 0.5f);
}

float3 OctahedralDirection(float2 uv)
{
    float px = uv.x * 2.0f - 1.0f;
    float py = uv.y * 2.0f - 1.0f;
    float pz = 1.0f - abs(px) - abs(py);
    if (pz < 0.0f)
    {
        float fx = (1.0f - abs(py)) * (px >= 0.0f ? 1.0f : -1.0f);
        float fy = (1.0f - abs(px)) * (py >= 0.0f ? 1.0f : -1.0f);
        px = fx;
        py = fy;
    }
    return normalize(float3(px, py, pz));
}

// The depth z of the distance t along a texel's ray.
float MomentDepth(MomentShadowMap m, float t)
{
    return (t - m.depthNear) / (m.depthFar - m.depthNear) * 2.0f - 1.0f;
}

// The ray of the map's texel at texture coordinates uv: from o along the unit vector d; the distance
// along it is what MomentDepth maps.
void MomentMapRay(MomentShadowMap m, float2 uv, AIR_OUT(float3) o, AIR_OUT(float3) d)
{
    if (m.kind == kMomentMapSun)
    {
        o = m.origin + m.axisU * uv.x + m.axisV * uv.y;
        d = m.axisW;
    }
    else
    {
        o = m.origin;
        d = OctahedralDirection(uv);
    }
}

// Where world point p lies in map m: texture coordinates and depth z (below -1: in front of
// everything the map holds, above 1 behind it). False beside a sun map.
bool MomentMapLocate(MomentShadowMap m, float3 p, AIR_OUT(float2) uv, AIR_OUT(float) z)
{
    float3 d = p - m.origin;
    float t;
    if (m.kind == kMomentMapSun)
    {
        uv = float2(dot(d, m.axisU) / dot(m.axisU, m.axisU), dot(d, m.axisV) / dot(m.axisV, m.axisV));
        t = dot(d, m.axisW);
    }
    else
    {
        t = length(d);
        uv = OctahedralUv(t > 0.0f ? d / t : float3(0.0f, 0.0f, 1.0f));
    }
    z = MomentDepth(m, t);
    return uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f;
}

// A step of optical depth dTau at depth z.
void AddMoments(AIR_INOUT(float) b0, AIR_INOUT(float4) b, float dTau, float z)
{
    float z2 = z * z;
    b0 += dTau;
    b = b + float4(z, z2, z2 * z, z2 * z2) * dTau;
}

// The share (0..1) of the optical depth in front of depth z, from the normalized moments m (b_k / b0).
float MomentAbsorbanceShare(float4 m, float z, float bias, float overestimation)
{
    // Pulled towards the moments of a distribution well inside the valid set: 0.625 at 0 and 0.1875 at
    // -1 and 1 (Muenstermann et al.'s vector for 32-bit moments).
    float b1 = lerp(m.x, 0.0f, bias);
    float b2 = lerp(m.y, 0.375f, bias);
    float b3 = lerp(m.z, 0.0f, bias);
    float b4 = lerp(m.w, 0.375f, bias);
    // An LDL^T factorization of the Hankel matrix [1 b1 b2; b1 b2 b3; b2 b3 b4].
    float d11 = b2 - b1 * b1;
    float l21d11 = b3 - b1 * b2;
    float l21 = l21d11 / d11;
    float d22 = (b4 - b2 * b2) - l21d11 * l21;
    // c solves B c = (1, z, z^2): the polynomial c0 + c1 x + c2 x^2 vanishes at the other two points.
    float c1 = z - b1;
    float c2 = z * z - b2 - l21 * c1;
    c1 /= d11;
    c2 /= d22;
    c1 -= l21 * c2;
    float c0 = 1.0f - (c1 * b1 + c2 * b2);
    float p = c1 / c2;
    float q = c0 / c2;
    float r = sqrt(max(p * p * 0.25f - q, 0.0f));
    float z1 = -p * 0.5f - r;
    float z2 = -p * 0.5f + r;
    // The weights of the three points, summed with the factors f: the quadratic through (z, f0),
    // (z1, f1) and (z2, f2) in Newton's form, whose expectation under the moments is that sum.
    float f0 = overestimation;
    float f1 = z1 < z ? 1.0f : 0.0f;
    float f2 = z2 < z ? 1.0f : 0.0f;
    float f01 = (f1 - f0) / (z1 - z);
    float f12 = (f2 - f1) / (z2 - z1);
    float f012 = (f12 - f01) / (z2 - z);
    // f0 + f01 (x - z) + f012 (x - z)(x - z1) = a0 + a1 x + a2 x^2
    float a2 = f012;
    float a1 = f01 - f012 * (z + z1);
    float a0 = f0 - f01 * z + f012 * z * z1;
    float share = a0 + a1 * b1 + a2 * b2;
    return isfinite(share) ? saturate(share) : (z > b1 ? 1.0f : 0.0f);
}

// The transmittance at depth z along a ray whose optical depth is b0 with raw moments b.
float MomentTransmittance(float b0, float4 b, float z)
{
    if (b0 < kMomentEmpty || z <= -1.0f)
    {
        return 1.0f;
    }
    if (z >= 1.0f)
    {
        return exp(-b0);
    }
    float share = MomentAbsorbanceShare(b / b0, z, kMomentBias, kMomentOverestimation);
    return saturate(exp(-b0 * share));
}

// How much of a light reaches world point p through the volumes of map m: one fetch of b0, and of
// the other moments only when the texel's ray met a volume.
float MomentShadow(MomentShadowMap m, float3 p)
{
    float2 uv;
    float z;
    if (!MomentMapLocate(m, p, uv, z) || z <= -1.0f)
    {
        return 1.0f;
    }
    float b0 = MomentMapB0Sample(m, uv);
    if (b0 < kMomentEmpty)
    {
        return 1.0f;
    }
    return MomentTransmittance(b0, MomentMapBSample(m, uv), z);
}

#endif
