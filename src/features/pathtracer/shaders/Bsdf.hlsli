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
// A material may wear a clear coat (BsdfMaterial::coat, docs/architecture.md §8.17): a rough
// dielectric interface of index kCoatIor over a layer that absorbs (coatColor: what it lets through
// at normal incidence, one way) over the material above. Its reflection off the interface is exact;
// the light that enters and leaves again (the inner part) has no closed form, and is
//   kCoatRandomWalk   estimated by a random walk between the interface and the base with next event
//                     estimation to wi (the position-free layered BSDF of Guo et al. 2018 as pbrt-v4's
//                     LayeredBxDF evaluates it): unbiased, noisy, the reference
//   kCoatApproximate  a single bounce through a smooth interface with a multiple-bounce factor
//   kCoatNeural       the approximation times a small MLP's correction learnt from the walk
//                     (NeuralCoat.hlsli)
// Coats are compiled with AIR_BSDF_COATS (their code costs kernels that do not need it, even unused);
// without, a coated material is its base.
// and the sampling picks the interface's reflection or the base's lobes (a proxy for the inner part:
// any of the three is weighed by the same pdf).
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

// The clear coat (BsdfMaterial::coat): how its inner part is evaluated.
AIR_CONST uint kCoatNone = 0u;
AIR_CONST uint kCoatNeural = 1u;
AIR_CONST uint kCoatRandomWalk = 2u;
AIR_CONST uint kCoatApproximate = 3u;
AIR_CONST float kCoatIor = 1.5f;
// Scattering events of a walk inside the layer, at most: light the interface traps (total internal
// reflection) keeps much of its weight for several round trips, which a short limit would cut off (10:
// 8% of a white coated base lost); Russian roulette ends most walks long before.
AIR_CONST uint kCoatWalkDepth = 64u;
// Hemispherical average of the interface's reflectance seen from inside, 1 - (1 - 0.092) / 1.5^2:
// what the approximation sends back down at every bounce off a diffuse base.
AIR_CONST float kCoatInternalReflectance = 0.596f;
#ifndef AIR_BSDF_COATS
#define AIR_BSDF_COATS 0
#endif

struct BsdfMaterial
{
    float3 baseColor;
    float roughness; // perceptual, alpha = roughness^2
    float metallic;
    uint coat;           // kCoat*: none, or how the coat's inner part is evaluated
    float coatRoughness; // perceptual, of the coat's interface
    float3 coatColor;    // what the coat's layer lets through at normal incidence, one way
    uint coatSeed;       // kCoatRandomWalk: the walks' random numbers at this shading point and frame
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

// Unpolarized Fresnel reflectance of a dielectric of relative index eta (cosI < 0: from inside).
float FrDielectric(float cosI, float eta)
{
    cosI = clamp(cosI, -1.0f, 1.0f);
    if (cosI < 0.0f)
    {
        eta = 1.0f / eta;
        cosI = -cosI;
    }
    float sin2T = (1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f)
    {
        return 1.0f;
    }
    float cosT = sqrt(max(0.0f, 1.0f - sin2T));
    float parallel = (eta * cosI - cosT) / (eta * cosI + cosT);
    float perpendicular = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (parallel * parallel + perpendicular * perpendicular);
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
    // With a coat: its interface's alpha, and the probability of sampling its reflection (otherwise
    // the base's lobes above).
    float coatAlpha;
    float coatProbability;
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
    c.coatAlpha = RoughnessToAlpha(m.coatRoughness);
    c.coatProbability = 0.0f;
    if (AIR_BSDF_COATS != 0 && m.coat != kCoatNone)
    {
        // The interface reflects F(wo) of the light; the base what passes twice through the layer.
        float f = FrDielectric(wo.z, kCoatIor);
        float3 through = m.coatColor * m.coatColor;
        float inner = (1.0f - f) * max(Luminance(m.baseColor * through), 0.02f);
        c.coatProbability = clamp(f / (f + inner), 0.1f, 0.9f);
    }
    return c;
}

// metal: GGX with Schlick(baseColor) plus its multiple scattering; dielectric: GGX with
// Schlick(0.04) plus multiple scattering plus the layered Lambert base; blended by metallic. The
// material without its coat.
BsdfEval EvaluateBaseBsdf(BsdfContext c, float3 wi)
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

// A direction drawn from the base's lobes (EvaluateBaseBsdf's pdf).
float3 SampleBaseDirection(BsdfContext c, float uLobe, float2 u)
{
    if (uLobe < c.specularProbability)
    {
        float3 h = SampleGgxVndf(c.wo, c.alpha, u);
        return reflect(-c.wo, h);
    }
    return SampleCosineHemisphere(u);
}

// --- The coat's interface -------------------------------------------------------------------------
// A rough dielectric (GGX, visible normal sampling) of index eta seen from either side, as pbrt-v4's
// DielectricBxDF: z > 0 is outside, values are f without the cosine, and `radiance` selects the
// transport mode (importance: without the 1 / eta'^2 of radiance crossing into another medium).

AIR_CONST uint kInterfaceReflection = 1u;
AIR_CONST uint kInterfaceTransmission = 2u;

struct InterfaceSample
{
    float3 wi;
    float f; // without the cosine
    float pdf;
    bool valid;
};


// wi refracted through a surface of normal n (either side); etap: the relative index crossed.
bool RefractInterface(float3 wi, float3 n, float eta, AIR_OUT(float) etap, AIR_OUT(float3) wt)
{
    float cosI = dot(n, wi);
    if (cosI < 0.0f)
    {
        eta = 1.0f / eta;
        cosI = -cosI;
        n = -n;
    }
    etap = eta;
    wt = float3(0.0f, 0.0f, 0.0f);
    float sin2T = max(0.0f, 1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f)
    {
        return false;
    }
    float cosT = sqrt(1.0f - sin2T);
    wt = -wi / eta + n * (cosI / eta - cosT);
    return true;
}

// The visible normal density of w (either side).
float GgxVisiblePdf(float3 w, float3 wm, float alpha)
{
    return GgxG1(w, alpha) / abs(w.z) * GgxD(wm, alpha) * abs(dot(w, wm));
}

// The microfacet normal of a reflection or refraction from wo to wi, facing +z; false when there is
// none or a facet would be seen from behind.
bool InterfaceHalfVector(float3 wo, float3 wi, float eta, AIR_OUT(float3) wm, AIR_OUT(float) etap)
{
    bool reflection = wo.z * wi.z > 0.0f;
    etap = reflection ? 1.0f : (wo.z > 0.0f ? eta : 1.0f / eta);
    wm = wi * etap + wo;
    if (wo.z == 0.0f || wi.z == 0.0f || dot(wm, wm) == 0.0f)
    {
        return false;
    }
    wm = normalize(wm);
    wm = wm.z < 0.0f ? -wm : wm;
    return dot(wm, wi) * wi.z >= 0.0f && dot(wm, wo) * wo.z >= 0.0f;
}

float EvaluateInterface(float3 wo, float3 wi, float alpha, bool radiance)
{
    float3 wm;
    float etap;
    if (!InterfaceHalfVector(wo, wi, kCoatIor, wm, etap))
    {
        return 0.0f;
    }
    float fr = FrDielectric(dot(wo, wm), kCoatIor);
    float dg = GgxD(wm, alpha) * GgxG2(wo, wi, alpha);
    if (wo.z * wi.z > 0.0f)
    {
        return dg * fr / abs(4.0f * wi.z * wo.z);
    }
    float sum = dot(wi, wm) + dot(wo, wm) / etap;
    float ft = dg * (1.0f - fr) * abs(dot(wi, wm) * dot(wo, wm) / (sum * sum * wi.z * wo.z));
    return radiance ? ft / (etap * etap) : ft;
}

// The pdf of SampleInterface drawing wi when restricted to `lobes`.
float InterfacePdf(float3 wo, float3 wi, float alpha, uint lobes)
{
    float3 wm;
    float etap;
    if (!InterfaceHalfVector(wo, wi, kCoatIor, wm, etap))
    {
        return 0.0f;
    }
    float r = FrDielectric(dot(wo, wm), kCoatIor);
    float pr = (lobes & kInterfaceReflection) != 0u ? r : 0.0f;
    float pt = (lobes & kInterfaceTransmission) != 0u ? 1.0f - r : 0.0f;
    if (pr + pt <= 0.0f)
    {
        return 0.0f;
    }
    if (wo.z * wi.z > 0.0f)
    {
        return GgxVisiblePdf(wo, wm, alpha) / (4.0f * abs(dot(wo, wm))) * pr / (pr + pt);
    }
    float sum = dot(wi, wm) + dot(wo, wm) / etap;
    return GgxVisiblePdf(wo, wm, alpha) * abs(dot(wi, wm)) / (sum * sum) * pt / (pr + pt);
}

InterfaceSample SampleInterface(float3 wo, float alpha, float uLobe, float2 u, uint lobes, bool radiance)
{
    InterfaceSample s;
    s.wi = float3(0.0f, 0.0f, 1.0f);
    s.f = 0.0f;
    s.pdf = 0.0f;
    s.valid = false;
    float3 wm = SampleGgxVndf(wo.z < 0.0f ? -wo : wo, alpha, u);
    float r = FrDielectric(dot(wo, wm), kCoatIor);
    float pr = (lobes & kInterfaceReflection) != 0u ? r : 0.0f;
    float pt = (lobes & kInterfaceTransmission) != 0u ? 1.0f - r : 0.0f;
    if (pr + pt <= 0.0f)
    {
        return s;
    }
    if (uLobe < pr / (pr + pt))
    {
        s.wi = reflect(-wo, wm);
        if (wo.z * s.wi.z <= 0.0f)
        {
            return s;
        }
        s.pdf = GgxVisiblePdf(wo, wm, alpha) / (4.0f * abs(dot(wo, wm))) * pr / (pr + pt);
        s.f = GgxD(wm, alpha) * GgxG2(wo, s.wi, alpha) * r / abs(4.0f * s.wi.z * wo.z);
    }
    else
    {
        float etap;
        float3 wt;
        if (!RefractInterface(wo, wm, kCoatIor, etap, wt) || wo.z * wt.z >= 0.0f)
        {
            return s;
        }
        s.wi = wt;
        float sum = dot(wt, wm) + dot(wo, wm) / etap;
        s.pdf = GgxVisiblePdf(wo, wm, alpha) * abs(dot(wt, wm)) / (sum * sum) * pt / (pr + pt);
        float ft = (1.0f - r) * GgxD(wm, alpha) * GgxG2(wo, wt, alpha) *
                   abs(dot(wt, wm) * dot(wo, wm) / (sum * sum * wt.z * wo.z));
        s.f = radiance ? ft / (etap * etap) : ft;
    }
    s.valid = s.pdf > 0.0f && s.f > 0.0f;
    return s;
}

// --- The coat's inner part ------------------------------------------------------------------------

// The base under the coat: the material without it.
BsdfMaterial CoatBase(BsdfMaterial m)
{
    BsdfMaterial b = m;
    b.coat = kCoatNone;
    return b;
}

// What the layer lets through along w (either way).
float3 CoatTransmittance(BsdfMaterial m, float3 w)
{
    return pow(max(m.coatColor, float3(1e-6f, 1e-6f, 1e-6f)), 1.0f / max(abs(w.z), 1e-3f));
}

// w (outside, z > 0) refracted through a smooth interface, pointing up inside the layer.
float3 CoatRefracted(float3 w)
{
    float3 inside = float3(w.x / kCoatIor, w.y / kCoatIor, 0.0f);
    inside.z = sqrt(max(0.0f, 1.0f - inside.x * inside.x - inside.y * inside.y));
    return inside;
}

// kCoatApproximate: one bounce off the base between two smooth crossings, (1 - F(wo)) (1 - F(wi))
// f_base(wo', wi') T(wo') T(wi') / eta^2 (the radiance gained entering and lost leaving cancel but
// for the solid angle), times 1 / (1 - albedo * kCoatInternalReflectance) for the light the
// interface reflects back down, albedo the base's diffuse part (what a specular base reflects stays
// near the direction it came in, where the interface lets most of it out); times cos(theta_i).
float3 CoatInnerApproximate(BsdfContext c, float3 wi)
{
    BsdfMaterial base = CoatBase(c.material);
    float3 wo2 = CoatRefracted(c.wo);
    float3 wi2 = CoatRefracted(wi);
    float3 f = EvaluateBaseBsdf(PrepareBsdf(base, wo2), wi2).fCos / max(wi2.z, 1e-4f);
    float crossings = (1.0f - FrDielectric(c.wo.z, kCoatIor)) * (1.0f - FrDielectric(wi.z, kCoatIor));
    float3 through = CoatTransmittance(base, wo2) * CoatTransmittance(base, wi2);
    float3 albedo = c.material.coatColor * c.material.coatColor * base.baseColor * (1.0f - saturate(base.metallic));
    float3 bounces = float3(1.0f, 1.0f, 1.0f) / (float3(1.0f, 1.0f, 1.0f) - albedo * kCoatInternalReflectance);
    return f * through * bounces * (crossings / (kCoatIor * kCoatIor) * wi.z);
}

// kCoatRandomWalk: one random walk's estimate of the inner part, times cos(theta_i). The walk enters
// through the interface at wo, alternates between the base (where it connects to wi through a
// transmission presampled from wi, and through its own next direction, the two weighed by the power
// heuristic) and the interface from inside (where it reflects), and is absorbed by the layer on the
// way; Russian roulette ends it once its weight is low.
float3 CoatInnerWalk(BsdfContext c, float3 wi, AIR_INOUT(uint) seed)
{
    float3 f = float3(0.0f, 0.0f, 0.0f);
    BsdfMaterial base = CoatBase(c.material);
    float alpha = c.coatAlpha;
    float u0 = Rand(seed);
    InterfaceSample entry = SampleInterface(c.wo, alpha, u0, Rand2(seed), kInterfaceTransmission, true);
    float u1 = Rand(seed);
    InterfaceSample exit = SampleInterface(wi, alpha, u1, Rand2(seed), kInterfaceTransmission, false);
    if (!entry.valid || !exit.valid)
    {
        return f;
    }
    float3 beta = float3(1.0f, 1.0f, 1.0f) * (entry.f * abs(entry.wi.z) / entry.pdf);
    float3 w = entry.wi; // inside, going down
    bool atTop = true;
    for (uint depth = 0u; depth < kCoatWalkDepth; ++depth)
    {
        if (depth > 3u && MaxComponent(beta) < 0.25f)
        {
            float q = max(0.0f, 1.0f - MaxComponent(beta));
            if (Rand(seed) < q)
            {
                break;
            }
            beta = beta / (1.0f - q);
        }
        beta = beta * CoatTransmittance(base, w);
        atTop = !atTop;
        if (atTop)
        {
            // Back at the interface from inside: reflect down again.
            float u = Rand(seed);
            InterfaceSample r = SampleInterface(-w, alpha, u, Rand2(seed), kInterfaceReflection, true);
            if (!r.valid)
            {
                break;
            }
            beta = beta * (r.f * abs(r.wi.z) / r.pdf);
            w = r.wi;
        }
        else
        {
            // At the base: connect to wi through the presampled transmission...
            BsdfContext b = PrepareBsdf(base, -w);
            BsdfEval toExit = EvaluateBaseBsdf(b, -exit.wi);
            if (toExit.pdf > 0.0f)
            {
                float weight = PowerHeuristic(exit.pdf, toExit.pdf);
                f += beta * toExit.fCos * CoatTransmittance(base, exit.wi) * (weight * exit.f / exit.pdf);
            }
            // ...then scatter, and connect through the new direction.
            float uLobe = Rand(seed);
            float3 next = SampleBaseDirection(b, uLobe, Rand2(seed));
            if (next.z <= 0.0f)
            {
                break;
            }
            BsdfEval e = EvaluateBaseBsdf(b, next);
            if (e.pdf <= 0.0f)
            {
                break;
            }
            beta = beta * e.fCos / e.pdf;
            w = next;
            float fExit = EvaluateInterface(-w, wi, alpha, true);
            if (fExit > 0.0f)
            {
                float weight = PowerHeuristic(e.pdf, InterfacePdf(-w, wi, alpha, kInterfaceTransmission));
                f += beta * CoatTransmittance(base, w) * (fExit * weight);
            }
        }
    }
    return f * wi.z;
}

// The coat's reflection off its interface, times cos(theta_i).
float CoatReflectionFCos(BsdfContext c, float3 wi)
{
    return EvaluateInterface(c.wo, wi, c.coatAlpha, true) * wi.z;
}

// The pdf of sampling that reflection with visible normals.
float CoatReflectionPdf(BsdfContext c, float3 wi)
{
    float3 h = normalize(c.wo + wi);
    return GgxVndfReflectionPdf(c.wo, h, c.coatAlpha);
}

#if AIR_BSDF_COATS
#include "NeuralCoat.hlsli"

float3 CoatInner(BsdfContext c, float3 wi)
{
    if (c.material.coat == kCoatNeural)
    {
        return NeuralCoatInner(c, wi);
    }
    if (c.material.coat == kCoatRandomWalk)
    {
        uint seed = Pcg(c.material.coatSeed ^ Pcg(asuint(wi.x) ^ Pcg(asuint(wi.y))));
        return CoatInnerWalk(c, wi, seed);
    }
    return CoatInnerApproximate(c, wi);
}
#endif

BsdfEval EvaluateBsdf(BsdfContext c, float3 wi)
{
#if AIR_BSDF_COATS
    if (c.material.coat == kCoatNone)
#endif
    {
        return EvaluateBaseBsdf(c, wi);
    }
#if AIR_BSDF_COATS
    BsdfEval e;
    e.fCos = float3(0.0f, 0.0f, 0.0f);
    e.fCosDiffuse = float3(0.0f, 0.0f, 0.0f);
    e.pdf = 0.0f;
    if (c.wo.z <= 0.0f || wi.z <= 0.0f)
    {
        return e;
    }
    float3 inner = CoatInner(c, wi);
    e.fCos = inner + CoatReflectionFCos(c, wi);
    e.fCosDiffuse = inner * (1.0f - saturate(c.material.metallic));
    e.pdf = c.coatProbability * CoatReflectionPdf(c, wi) + (1.0f - c.coatProbability) * EvaluateBaseBsdf(c, wi).pdf;
    return e;
#endif
}

// A direction drawn from EvaluateBsdf's pdf (z <= 0: none).
float3 SampleBsdfDirection(BsdfContext c, float uLobe, float2 u)
{
    if (uLobe < c.coatProbability)
    {
        return reflect(-c.wo, SampleGgxVndf(c.wo, c.coatAlpha, u));
    }
    float rest = (uLobe - c.coatProbability) / (1.0f - c.coatProbability);
    return SampleBaseDirection(c, min(rest, 0.99999994f), u);
}

BsdfSample InvalidBsdfSample()
{
    BsdfSample s;
    s.wi = float3(0.0f, 0.0f, 1.0f);
    s.weight = 0.0f;
    s.diffuseWeight = 0.0f;
    s.pdf = 0.0f;
    s.valid = false;
    return s;
}

// wi with the BSDF's weight for it (e: EvaluateBsdf there).
BsdfSample MakeBsdfSample(float3 wi, BsdfEval e)
{
    BsdfSample s = InvalidBsdfSample();
    if (e.pdf <= 0.0f)
    {
        return s;
    }
    s.wi = wi;
    s.pdf = e.pdf;
    s.weight = e.fCos / e.pdf;
    s.diffuseWeight = e.fCosDiffuse / e.pdf;
    s.valid = true;
    return s;
}

BsdfSample SampleBsdf(BsdfContext c, float uLobe, float2 u)
{
    if (c.wo.z <= 0.0f)
    {
        return InvalidBsdfSample();
    }
    float3 wi = SampleBsdfDirection(c, uLobe, u);
    if (wi.z <= 0.0f)
    {
        return InvalidBsdfSample();
    }
    return MakeBsdfSample(wi, EvaluateBsdf(c, wi));
}

BsdfEval EvaluateBsdf(BsdfMaterial m, float3 wo, float3 wi)
{
    return EvaluateBsdf(PrepareBsdf(m, wo), wi);
}

BsdfSample SampleBsdf(BsdfMaterial m, float3 wo, float uLobe, float2 u)
{
    return SampleBsdf(PrepareBsdf(m, wo), uLobe, u);
}

// --- On a surface ---------------------------------------------------------------------------------
// The BSDF where the shading normal (z, interpolated or bent by a normal map) leans away from the
// surface's geometric normal ng (local, on wo's side). The diffuse lobe, a cosine around the shading
// normal, then reaches below the surface, where no light comes from: that part is mirrored in the
// surface's plane back above it (the way Schuessler et al. 2017 reflect light off a facet of the
// microsurface; here the facet is the surface itself). A direction above the surface gets the diffuse
// lobe's value and pdf at its mirror image too, and a diffuse sample below the surface is mirrored, so
// sampling, evaluation and MIS agree. Lost before: 1.8% of `textured` in a white furnace, 0.2% of
// `sphere_field` and `terrain` (interpolated normals). The specular lobes' share below the surface is
// small and still lost: mirroring them takes a second evaluation of the whole BSDF, which cost every
// kernel 5-7% of its frame in registers (17% with coats), also where nothing is mirrored. Coated
// materials are left as they are.

float3 MirrorInSurface(float3 w, float3 ng)
{
    return w - 2.0f * dot(w, ng) * ng;
}

// The diffuse lobe of EvaluateBaseBsdf alone at wi (z > 0), and its share of the sampling pdf.
BsdfEval EvaluateBaseDiffuse(BsdfContext c, float3 wi)
{
    float esi = LutBilinear(kLutEs, wi.z, c.roughness);
    float3 diffuse = c.material.baseColor *
                     ((1.0f - c.eso) * (1.0f - esi) * c.diffuseScale * wi.z * (1.0f - saturate(c.material.metallic)));
    BsdfEval e;
    e.fCos = diffuse;
    e.fCosDiffuse = diffuse;
    e.pdf = (1.0f - c.specularProbability) * wi.z * kInvPi;
    return e;
}

bool FoldsDiffuse(BsdfContext c)
{
    return c.coatProbability == 0.0f;
}

BsdfEval EvaluateSurfaceBsdf(BsdfContext c, float3 ng, float3 wi)
{
    BsdfEval e = EvaluateBsdf(c, wi);
    float3 mirrored = MirrorInSurface(wi, ng);
    if (FoldsDiffuse(c) && mirrored.z > 0.0f && dot(wi, ng) > 0.0f && c.wo.z > 0.0f)
    {
        BsdfEval m = EvaluateBaseDiffuse(c, mirrored);
        e.fCos += m.fCos;
        e.fCosDiffuse += m.fCosDiffuse;
        e.pdf += m.pdf;
    }
    return e;
}

BsdfSample SampleSurfaceBsdf(BsdfContext c, float3 ng, float uLobe, float2 u)
{
    if (c.wo.z <= 0.0f)
    {
        return InvalidBsdfSample();
    }
    float3 wi = SampleBsdfDirection(c, uLobe, u);
    if (wi.z <= 0.0f)
    {
        return InvalidBsdfSample();
    }
    if (dot(wi, ng) <= 0.0f)
    {
        // Mirrored when the diffuse lobe drew it (SampleBaseDirection's choice), lost otherwise.
        bool diffuse = FoldsDiffuse(c) && uLobe >= c.specularProbability;
        wi = MirrorInSurface(wi, ng);
        if (!diffuse || dot(wi, ng) <= 0.0f)
        {
            return InvalidBsdfSample();
        }
    }
    return MakeBsdfSample(wi, EvaluateSurfaceBsdf(c, ng, wi));
}

BsdfEval EvaluateSurfaceBsdf(BsdfMaterial m, float3 wo, float3 ng, float3 wi)
{
    return EvaluateSurfaceBsdf(PrepareBsdf(m, wo), ng, wi);
}

BsdfSample SampleSurfaceBsdf(BsdfMaterial m, float3 wo, float3 ng, float uLobe, float2 u)
{
    return SampleSurfaceBsdf(PrepareBsdf(m, wo), ng, uLobe, u);
}

// The directional albedos towards wo of the light the BSDF splits off as diffuse (fCosDiffuse) and
// of the rest, for the denoisers' guides (they divide each lobe's light by its albedo and multiply
// the filtered result by it again) and debug views. They must follow the lobes as the BSDF weighs
// them: the diffuse lobe gives up (1 - Es(wo)) to the specular, and a coat's inner part (counted as
// diffuse but for its metallic share) loses what the interface reflects. A guide that does not divides
// the light unevenly at grazing angles, where the denoisers' filters then mix unlike values.
void BaseAlbedos(BsdfMaterial m, float muO, AIR_OUT(float3) diffuse, AIR_OUT(float3) specular)
{
    float roughness = max(m.roughness, kMinRoughness);
    float e = LutBilinear(kLutE, muO, roughness);
    float es = LutBilinear(kLutEs, muO, roughness);
    float3 metal = FresnelSchlick(m.baseColor, muO) * e + MultiScatterFresnel(m.baseColor, roughness) * (1.0f - e);
    diffuse = m.baseColor * ((1.0f - saturate(m.metallic)) * (1.0f - es));
    specular = lerp(float3(es, es, es), metal, saturate(m.metallic));
}

void GuideAlbedos(BsdfMaterial m, float muO, AIR_OUT(float3) diffuse, AIR_OUT(float3) specular)
{
    muO = saturate(muO);
    BaseAlbedos(m, muO, diffuse, specular);
#if AIR_BSDF_COATS
    if (m.coat != kCoatNone)
    {
        // CoatInnerApproximate's view: enter at wo (refracted, absorbed on the way down), meet the
        // base, leave through the interface or bounce back down from it (kCoatInternalReflectance).
        BsdfMaterial base = CoatBase(m);
        float3 wo2 = CoatRefracted(float3(sqrt(max(1.0f - muO * muO, 0.0f)), 0.0f, muO));
        float3 baseDiffuse;
        float3 baseSpecular;
        BaseAlbedos(base, wo2.z, baseDiffuse, baseSpecular);
        float3 a = baseDiffuse + baseSpecular;
        float3 up = m.coatColor * m.coatColor; // a diffuse way out crosses the layer at 1 / cos = 2 on average
        float3 inner = (1.0f - FrDielectric(muO, kCoatIor)) * CoatTransmittance(base, wo2) * a * up *
                       (1.0f - kCoatInternalReflectance) /
                       (float3(1.0f, 1.0f, 1.0f) - a * up * kCoatInternalReflectance);
        float f = FrDielectric(muO, kCoatIor);
        diffuse = inner * (1.0f - saturate(m.metallic));
        specular = float3(f, f, f) + inner * saturate(m.metallic);
    }
#endif
}

#endif
