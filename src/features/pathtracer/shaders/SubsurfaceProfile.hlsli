// The profile half of subsurface scattering (Subsurface.hlsli, docs/architecture.md §8.20): the medium
// beneath a subsurface material and the light leaving a flat half-space of it at a distance r from where
// it entered, by Burley's normalized diffusion and by the neural profile. Needs only Bsdf.hlsli's LUT,
// so the CPU's fit of the network (SubsurfaceFit.cpp) compiles it too. Shared by HLSL and C++.
#ifndef AIR_PT_SUBSURFACE_PROFILE_HLSLI
#define AIR_PT_SUBSURFACE_PROFILE_HLSLI

#include "SubsurfaceProfileNetwork.h"

AIR_CONST uint kSubsurfaceNeural = 0u;
AIR_CONST uint kSubsurfaceRandomWalk = 1u;
AIR_CONST uint kSubsurfaceApproximate = 2u;
AIR_CONST float kSubsurfaceMinRadius = 1e-5f; // metres: a channel without spread leaves at once

struct SubsurfaceMedium
{
    float3 color;  // what a half-space reflects in all (baseColor), A
    float3 radius; // metres, d
    float3 sigmaT; // per metre
    float3 albedo; // single scattering
    float g;
};

float SubsurfaceComponent(float3 v, uint c)
{
    return c == 0u ? v.x : (c == 1u ? v.y : v.z);
}

float SubsurfaceMean(float3 v)
{
    return (v.x + v.y + v.z) * (1.0f / 3.0f);
}

// The single-scattering albedo whose half-space reflects `color` in all (the BSDF LUT's inversion of
// Subsurface.hlsli's walk in a flat half-space, between its rows of g).
float SubsurfaceSingleScattering(float color, float g)
{
    float fg = saturate(g / 0.9f) * float(kSubsurfaceLutRows - 1u);
    uint g0 = min(uint(fg), kSubsurfaceLutRows - 2u);
    float t = fg - float(g0);
    float s0 = LutLinear(kLutSubsurface + g0 * kBsdfLutSize, color);
    float s1 = LutLinear(kLutSubsurface + (g0 + 1u) * kBsdfLutSize, color);
    float s = lerp(s0, s1, t); // sqrt(1 - albedo)
    return 1.0f - s * s;
}

// The extinction per unit of the radius: Chiang et al. 2016's scale (the spread of an isotropic medium
// of extinction 1 / (radius s(A)) is about radius), divided by 1 - g so that the reduced extinction
// sigma_t (1 - g), which sets how far light diffuses, stays the same whatever the anisotropy (Burley's
// profile then fits every g alike; without, g = 0.9 spread ten times as far as its radius).
float SubsurfaceExtinction(float color, float g)
{
    float a = color - 0.8f;
    return 1.0f / ((1.9f - color + 3.5f * a * a) * (1.0f - min(g, 0.9f)));
}

SubsurfaceMedium MakeSubsurfaceMedium(float3 color, float3 radius, float g)
{
    SubsurfaceMedium m;
    m.color = float3(saturate(color.x), saturate(color.y), saturate(color.z));
    m.radius = max(radius, float3(kSubsurfaceMinRadius, kSubsurfaceMinRadius, kSubsurfaceMinRadius));
    m.sigmaT = float3(SubsurfaceExtinction(m.color.x, g) / m.radius.x, SubsurfaceExtinction(m.color.y, g) / m.radius.y,
                      SubsurfaceExtinction(m.color.z, g) / m.radius.z);
    m.albedo = float3(SubsurfaceSingleScattering(m.color.x, g), SubsurfaceSingleScattering(m.color.y, g),
                      SubsurfaceSingleScattering(m.color.z, g));
    m.g = g;
    return m;
}

// Burley's normalized diffusion over the plane, per unit of the colour: (e^(-r/d) + e^(-r/3d)) / (8 pi d r),
// which integrates to 1.
float BurleyProfile(float r, float d)
{
    r = max(r, 1e-6f * d);
    return (exp(-r / d) + exp(-r / (3.0f * d))) / (8.0f * kPi * d * r);
}

// The pdf over r of sampling it (2 pi r BurleyProfile): a quarter e^(-r/d) / d, three quarters
// e^(-r/3d) / 3d.
float BurleyRadialPdf(float r, float d)
{
    return (exp(-r / d) + exp(-r / (3.0f * d))) / (4.0f * d);
}

float SampleBurleyRadius(float d, float u)
{
    if (u < 0.25f)
    {
        return -d * log(1.0f - u * 4.0f);
    }
    return -3.0f * d * log(1.0f - (u - 0.25f) * (4.0f / 3.0f));
}

// The network's correction of Burley's profile for one channel: z(log(r / d), A, g), |z| <= 4.
float SubsurfaceProfileCorrection(float r, float d, float color, float g)
{
    float x[kSubsurfaceProfileInputs];
    x[0] = log(max(r / d, 1e-4f)) * 0.25f;
    x[1] = color * 2.0f - 1.0f;
    x[2] = g * (2.0f / 0.9f) - 1.0f;
    // Unrolled: the weights' indices are constants (immediates in the shader) and the activations stay
    // in registers; looped, the arrays went to memory and the network cost 2.6 times the trace.
    float h1[kSubsurfaceProfileHidden];
    uint w = 0u;
    AIR_UNROLL
    for (uint i = 0u; i < kSubsurfaceProfileHidden; ++i)
    {
        float a = kSubsurfaceProfileWeights[w + kSubsurfaceProfileInputs];
        AIR_UNROLL
        for (uint k = 0u; k < kSubsurfaceProfileInputs; ++k)
        {
            a += kSubsurfaceProfileWeights[w + k] * x[k];
        }
        h1[i] = max(a, 0.0f);
        w += kSubsurfaceProfileInputs + 1u;
    }
    float h2[kSubsurfaceProfileHidden];
    AIR_UNROLL
    for (uint i = 0u; i < kSubsurfaceProfileHidden; ++i)
    {
        float a = kSubsurfaceProfileWeights[w + kSubsurfaceProfileHidden];
        AIR_UNROLL
        for (uint k = 0u; k < kSubsurfaceProfileHidden; ++k)
        {
            a += kSubsurfaceProfileWeights[w + k] * h1[k];
        }
        h2[i] = max(a, 0.0f);
        w += kSubsurfaceProfileHidden + 1u;
    }
    float z = kSubsurfaceProfileWeights[w + kSubsurfaceProfileHidden];
    AIR_UNROLL
    for (uint k = 0u; k < kSubsurfaceProfileHidden; ++k)
    {
        z += kSubsurfaceProfileWeights[w + k] * h2[k];
    }
    return clamp(z, -4.0f, 4.0f);
}

// R(r) of each channel: the light leaving per unit area at distance r per unit that entered.
float3 SubsurfaceProfile(SubsurfaceMedium m, float r, uint model)
{
    float3 profile = m.color * float3(BurleyProfile(r, m.radius.x), BurleyProfile(r, m.radius.y),
                                      BurleyProfile(r, m.radius.z));
    if (model == kSubsurfaceNeural)
    {
        profile *= float3(exp(SubsurfaceProfileCorrection(r, m.radius.x, m.color.x, m.g)),
                          exp(SubsurfaceProfileCorrection(r, m.radius.y, m.color.y, m.g)),
                          exp(SubsurfaceProfileCorrection(r, m.radius.z, m.color.z, m.g)));
    }
    return profile;
}

#endif
