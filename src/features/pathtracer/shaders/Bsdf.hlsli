// Energy-conserving metallic / dielectric BSDF. Shared by HLSL and C++.
//
// Model (all lobes reciprocal, white furnace exact up to LUT resolution):
//   metal      f = GGX(F = Schlick(baseColor)) + Kulla-Conty multiple scattering (colored by Fms)
//   dielectric f = GGX(F = Schlick(0.04)) + Kulla-Conty multiple scattering
//                + Lambert(baseColor) * (1 - Es(wo)) (1 - Es(wi)) / (1 - EsAvg)
//   material   f = lerp(dielectric, metal, metallic)
// GGX uses Smith height-correlated masking-shadowing and is sampled with visible normals.
// The dielectric layering term (Kelemen & Szirmay-Kalos style) hands the diffuse base exactly the
// energy the specular coat did not reflect, so a white dielectric reflects 100% at every angle.
//
// E(mu, r)     directional albedo of single-scatter GGX with F = 1
// Eavg(r)      2 * integral E(mu) mu dmu
// Es(mu, r)    directional albedo of the full dielectric specular lobe (F0 = 0.04, with MS)
// EsAvg(r)     2 * integral Es(mu) mu dmu
// are tabulated by BuildBsdfLut() (C++, using these same functions) and read through
// BsdfLutLoad(index), which the including code must define. r is the perceptual roughness.
//
// Directions are in the local shading frame (z = normal) and point away from the surface.
#ifndef AIR_PT_BSDF_HLSLI
#define AIR_PT_BSDF_HLSLI

#include "Sampling.hlsli"

AIR_CONST uint kBsdfLutSize = 32;
AIR_CONST uint kLutE = 0;                                          // [r][mu]
AIR_CONST uint kLutEavg = kBsdfLutSize * kBsdfLutSize;             // [r]
AIR_CONST uint kLutEs = kLutEavg + kBsdfLutSize;                   // [r][mu]
AIR_CONST uint kLutEsAvg = kLutEs + kBsdfLutSize * kBsdfLutSize;   // [r]
AIR_CONST uint kBsdfLutFloats = kLutEsAvg + kBsdfLutSize;
AIR_CONST float kDielectricF0 = 0.04f;
AIR_CONST float kMinRoughness = 0.03f; // alpha >= 9e-4: keeps D finite in float

struct BsdfMaterial
{
    float3 baseColor;
    float roughness; // perceptual, alpha = roughness^2
    float metallic;
};

struct BsdfEval
{
    float3 fCos;        // f(wo, wi) * cos(theta_i)
    float3 fCosDiffuse; // the part of fCos from the diffuse base (the rest is specular), for denoisers
    float pdf;          // solid angle pdf of SampleBsdf producing wi
};

struct BsdfSample
{
    float3 wi;
    float3 weight;        // f * cos / pdf
    float3 diffuseWeight; // the diffuse base's part of it
    float pdf;
    bool valid;
};

// --- LUT access --------------------------------------------------------------------------------

float LutLinear(uint base, float x)
{
    float fx = saturate(x) * float(kBsdfLutSize - 1);
    uint i0 = min(uint(fx), kBsdfLutSize - 2);
    float t = fx - float(i0);
    return lerp(BsdfLutLoad(base + i0), BsdfLutLoad(base + i0 + 1), t);
}

float LutBilinear(uint base, float mu, float roughness)
{
    float fr = saturate(roughness) * float(kBsdfLutSize - 1);
    uint r0 = min(uint(fr), kBsdfLutSize - 2);
    float t = fr - float(r0);
    return lerp(LutLinear(base + r0 * kBsdfLutSize, mu), LutLinear(base + (r0 + 1) * kBsdfLutSize, mu), t);
}

// --- GGX -----------------------------------------------------------------------------------------

float GgxD(float3 h, float alpha)
{
    float a2 = alpha * alpha;
    float d = h.z * h.z * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d);
}

float GgxLambda(float3 w, float alpha)
{
    float z2 = w.z * w.z;
    float tan2 = max(0.0f, 1.0f - z2) / max(z2, 1e-12f);
    return 0.5f * (sqrt(1.0f + alpha * alpha * tan2) - 1.0f);
}

float GgxG1(float3 w, float alpha)
{
    return 1.0f / (1.0f + GgxLambda(w, alpha));
}

// Smith height-correlated masking-shadowing.
float GgxG2(float3 wo, float3 wi, float alpha)
{
    return 1.0f / (1.0f + GgxLambda(wo, alpha) + GgxLambda(wi, alpha));
}

// Visible normal sampling with spherical caps (Dupuy & Benyoub 2023). wo.z > 0.
float3 SampleGgxVndf(float3 wo, float alpha, float2 u)
{
    float3 woStd = normalize(float3(wo.x * alpha, wo.y * alpha, wo.z));
    float phi = kTwoPi * u.x;
    float z = (1.0f - u.y) * (1.0f + woStd.z) - woStd.z;
    float sinTheta = sqrt(saturate(1.0f - z * z));
    float3 c = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    float3 hStd = c + woStd;
    return normalize(float3(hStd.x * alpha, hStd.y * alpha, max(hStd.z, 1e-6f)));
}

// pdf of wi = reflect(-wo, h) when h is drawn with SampleGgxVndf.
float GgxVndfReflectionPdf(float3 wo, float3 h, float alpha)
{
    return GgxG1(wo, alpha) * GgxD(h, alpha) / (4.0f * wo.z);
}

float3 FresnelSchlick(float3 f0, float cosTheta)
{
    float m = saturate(1.0f - cosTheta);
    float m2 = m * m;
    return f0 + (1.0f - f0) * (m2 * m2 * m);
}

// Hemispherical average of Schlick's Fresnel: 2 * integral F(mu) mu dmu = f0 + (1 - f0) / 21.
float3 FresnelAverage(float3 f0)
{
    return f0 + (1.0f - f0) * (1.0f / 21.0f);
}

float RoughnessToAlpha(float roughness)
{
    float r = max(roughness, kMinRoughness);
    return r * r;
}

// Single-scatter GGX reflection with Fresnel = 1, times cos(theta_i).
float GgxSingleScatterCos(float3 wo, float3 wi, float alpha)
{
    float3 h = normalize(wo + wi);
    return GgxD(h, alpha) * GgxG2(wo, wi, alpha) / (4.0f * wo.z);
}

// Color of the multiple-scattering lobe: Favg^2 Eavg / (1 - Favg (1 - Eavg)) (Kulla & Conty 2017).
float3 MultiScatterFresnelFromAverage(float3 f0, float eavg)
{
    float3 favg = FresnelAverage(f0);
    return favg * favg * eavg / (1.0f - favg * (1.0f - eavg));
}

float3 MultiScatterFresnel(float3 f0, float roughness)
{
    return MultiScatterFresnelFromAverage(f0, LutLinear(kLutEavg, roughness));
}

// --- Full material -------------------------------------------------------------------------------

// The parts of a shading point's BSDF that depend only on the material and wo: most of the LUT
// lookups (4 dependent loads each). Prepared once per point and shared by every evaluation and
// sample there, which then only look up E(wi) and Es(wi).
struct BsdfContext
{
    BsdfMaterial material;
    float3 wo;
    float roughness; // clamped to kMinRoughness
    float alpha;
    float eo;           // E(wo)
    float eso;          // Es(wo)
    float msScale;      // 1 / (pi (1 - Eavg)): Kulla-Conty lobe normalization
    float diffuseScale; // 1 / (pi (1 - EsAvg)): layered diffuse normalization
    float3 metalMsFresnel;
    float3 dielectricMsFresnel;
    // Probability of sampling the GGX lobe (otherwise the cosine lobe), proportional to the energy
    // the specular lobes reflect towards wo.
    float specularProbability;
};

BsdfContext PrepareBsdf(BsdfMaterial m, float3 wo)
{
    BsdfContext c;
    c.material = m;
    c.wo = wo;
    c.roughness = max(m.roughness, kMinRoughness);
    c.alpha = c.roughness * c.roughness;
    c.eo = LutBilinear(kLutE, wo.z, c.roughness);
    c.eso = LutBilinear(kLutEs, wo.z, c.roughness);
    float eavg = LutLinear(kLutEavg, c.roughness);
    float esAvg = LutLinear(kLutEsAvg, c.roughness);
    c.msScale = 1.0f / (kPi * max(1.0f - eavg, 1e-4f));
    c.diffuseScale = kInvPi / max(1.0f - esAvg, 1e-4f);
    c.metalMsFresnel = MultiScatterFresnelFromAverage(m.baseColor, eavg);
    c.dielectricMsFresnel = MultiScatterFresnelFromAverage(float3(kDielectricF0, kDielectricF0, kDielectricF0), eavg);
    float diffuseEnergy = (1.0f - c.eso) * max(Luminance(m.baseColor), 0.0f);
    float pDielectric = c.eso + diffuseEnergy > 0.0f ? c.eso / (c.eso + diffuseEnergy) : 1.0f;
    float pMetal = c.eo; // the rest is the diffuse-like multiple-scattering lobe
    c.specularProbability = saturate(lerp(pDielectric, pMetal, saturate(m.metallic)));
    return c;
}

// metal: GGX with Schlick(baseColor) plus its multiple scattering; dielectric: GGX with
// Schlick(0.04) plus multiple scattering plus the layered Lambert base; blended by metallic.
BsdfEval EvaluateBsdf(BsdfContext c, float3 wi)
{
    BsdfEval e;
    e.fCos = float3(0.0f, 0.0f, 0.0f);
    e.fCosDiffuse = float3(0.0f, 0.0f, 0.0f);
    e.pdf = 0.0f;
    float3 wo = c.wo;
    if (wo.z <= 0.0f || wi.z <= 0.0f)
    {
        return e;
    }
    float3 h = normalize(wo + wi);
    float ss = GgxSingleScatterCos(wo, wi, c.alpha);
    float ei = LutBilinear(kLutE, wi.z, c.roughness);
    float ms = (1.0f - c.eo) * (1.0f - ei) * c.msScale * wi.z;
    float cosOH = saturate(dot(wo, h));
    float3 dielectricF0 = kDielectricF0;
    float3 metal = FresnelSchlick(c.material.baseColor, cosOH) * ss + c.metalMsFresnel * ms;
    float3 dielectricSpecular = FresnelSchlick(dielectricF0, cosOH) * ss + c.dielectricMsFresnel * ms;
    float esi = LutBilinear(kLutEs, wi.z, c.roughness);
    float3 diffuse = c.material.baseColor * ((1.0f - c.eso) * (1.0f - esi) * c.diffuseScale * wi.z);
    e.fCos = lerp(dielectricSpecular + diffuse, metal, saturate(c.material.metallic));
    e.fCosDiffuse = diffuse * (1.0f - saturate(c.material.metallic));
    e.pdf = c.specularProbability * GgxVndfReflectionPdf(wo, h, c.alpha) +
            (1.0f - c.specularProbability) * wi.z * kInvPi;
    return e;
}

BsdfSample SampleBsdf(BsdfContext c, float uLobe, float2 u)
{
    BsdfSample s;
    s.wi = float3(0.0f, 0.0f, 1.0f);
    s.weight = 0.0f;
    s.diffuseWeight = 0.0f;
    s.pdf = 0.0f;
    s.valid = false;
    if (c.wo.z <= 0.0f)
    {
        return s;
    }
    if (uLobe < c.specularProbability)
    {
        float3 h = SampleGgxVndf(c.wo, c.alpha, u);
        s.wi = reflect(-c.wo, h);
    }
    else
    {
        s.wi = SampleCosineHemisphere(u);
    }
    if (s.wi.z <= 0.0f)
    {
        return s;
    }
    BsdfEval e = EvaluateBsdf(c, s.wi);
    if (e.pdf <= 0.0f)
    {
        return s;
    }
    s.pdf = e.pdf;
    s.weight = e.fCos / e.pdf;
    s.diffuseWeight = e.fCosDiffuse / e.pdf;
    s.valid = true;
    return s;
}

BsdfEval EvaluateBsdf(BsdfMaterial m, float3 wo, float3 wi)
{
    return EvaluateBsdf(PrepareBsdf(m, wo), wi);
}

BsdfSample SampleBsdf(BsdfMaterial m, float3 wo, float uLobe, float2 u)
{
    return SampleBsdf(PrepareBsdf(m, wo), uLobe, u);
}

// Directional albedo of the full material towards wo (for denoiser guides and debug views).
float3 DiffuseAlbedo(BsdfMaterial m)
{
    return m.baseColor * (1.0f - saturate(m.metallic));
}

float3 SpecularAlbedo(BsdfMaterial m, float muO)
{
    float roughness = max(m.roughness, kMinRoughness);
    float e = LutBilinear(kLutE, muO, roughness);
    float es = LutBilinear(kLutEs, muO, roughness);
    float3 metal = FresnelSchlick(m.baseColor, muO) * e + MultiScatterFresnel(m.baseColor, roughness) * (1.0f - e);
    return lerp(float3(es, es, es), metal, saturate(m.metallic));
}

#endif
