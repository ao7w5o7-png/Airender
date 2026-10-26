// Compiles shaders/Bsdf.hlsli as C++ and builds its lookup tables.

#include "air/features/pathtracer/Bsdf.h"

#include "air/renderer/shader/HlslCompat.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace air::pt
{
namespace shader
{

using namespace air::hlsl;

// The table BsdfLutLoad() reads. Points at the finished LUT, or at the partial one while it is built.
thread_local const float* tLut = nullptr;
// The neural coat's parameters (BindNeuralCoat); null: none, the network outputs no correction.
thread_local const float* tNeuralCoat = nullptr;

float BsdfLutLoad(uint index);
float4 NeuralCoatWeights4(uint index);

#define AIR_BSDF_COATS 1
#include "Bsdf.hlsli"

float BsdfLutLoad(uint index)
{
    return tLut[index];
}

float4 NeuralCoatWeights4(uint index)
{
    const float* p = tNeuralCoat;
    return p ? float4(p[index], p[index + 1], p[index + 2], p[index + 3]) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

} // namespace shader

namespace
{

using shader::float2;
using shader::float3;
using shader::kBsdfLutSize;

// Stratified sample grid per table entry: kGrid^2 visible normal samples.
constexpr u32 kGrid = 48;

float NodeValue(u32 i)
{
    return float(i) / float(kBsdfLutSize - 1);
}

// Directional albedo of single-scatter GGX, optionally with Schlick Fresnel (f0 >= 0).
float IntegrateSingleScatter(float mu, float roughness, float f0)
{
    const float alpha = shader::RoughnessToAlpha(roughness);
    const float3 wo(std::sqrt(1.0f - mu * mu), 0.0f, mu);
    const float g1o = shader::GgxG1(wo, alpha);
    double sum = 0.0;
    for (u32 y = 0; y < kGrid; ++y)
    {
        for (u32 x = 0; x < kGrid; ++x)
        {
            const float2 u((float(x) + 0.5f) / kGrid, (float(y) + 0.5f) / kGrid);
            const float3 h = shader::SampleGgxVndf(wo, alpha, u);
            const float3 wi = shader::reflect(-wo, h);
            if (wi.z <= 0.0f)
            {
                continue;
            }
            // f cos / pdf = G2 / G1(wo) for visible normal sampling.
            float weight = shader::GgxG2(wo, wi, alpha) / g1o;
            if (f0 >= 0.0f)
            {
                weight *= shader::FresnelSchlick(float3(f0), shader::saturate(shader::dot(wo, h))).x;
            }
            sum += weight;
        }
    }
    return float(sum / double(kGrid * kGrid));
}

// 2 * integral of mu * E(mu) over [0, 1] for E linear between the table nodes, i.e. exactly the
// cosine-weighted average of what LutLinear() returns. Keeps the multiple-scattering and diffuse
// lobes energy-consistent with the interpolated table.
float CosineAverage(const float* row)
{
    double sum = 0.0;
    for (u32 i = 0; i + 1 < kBsdfLutSize; ++i)
    {
        const double a = NodeValue(i);
        const double b = NodeValue(i + 1);
        sum += (b - a) / 6.0 * (row[i] * (2.0 * a + b) + row[i + 1] * (a + 2.0 * b));
    }
    return float(2.0 * sum);
}

// Subsurface scattering (Subsurface.hlsli): the light a half-space of the medium reflects in all, by
// Monte Carlo (400000 walks each: cosine-weighted entry through an index-matched boundary, Henyey-
// Greenstein scattering, the walk SubsurfaceWalk takes in a flat slab), for single-scattering albedos
// 1 - (1 - i / 31)^3 and g = 0, 0.3, 0.6, 0.9. Without absorption all of the light leaves.
constexpr float kSubsurfaceReflectance[shader::kSubsurfaceLutRows][kBsdfLutSize] = {
    {0.00000f, 0.02017f, 0.04135f, 0.06342f, 0.08655f, 0.11113f, 0.13667f, 0.16301f, 0.18996f, 0.21952f, 0.24943f,
     0.28017f, 0.31305f, 0.34620f, 0.38068f, 0.41730f, 0.45536f, 0.49344f, 0.53134f, 0.57316f, 0.61468f, 0.65650f,
     0.69843f, 0.73982f, 0.78169f, 0.82150f, 0.86204f, 0.89834f, 0.93234f, 0.96277f, 0.98671f, 1.0f},
    {0.00000f, 0.01378f, 0.02889f, 0.04499f, 0.06254f, 0.08128f, 0.10109f, 0.12298f, 0.14592f, 0.17065f, 0.19702f,
     0.22510f, 0.25553f, 0.28815f, 0.32092f, 0.35683f, 0.39309f, 0.43261f, 0.47249f, 0.51568f, 0.56060f, 0.60487f,
     0.65151f, 0.69853f, 0.74468f, 0.79147f, 0.83613f, 0.88030f, 0.92051f, 0.95578f, 0.98420f, 1.0f},
    {0.00000f, 0.00782f, 0.01639f, 0.02622f, 0.03685f, 0.04897f, 0.06218f, 0.07682f, 0.09314f, 0.11082f, 0.13087f,
     0.15379f, 0.17631f, 0.20454f, 0.23423f, 0.26614f, 0.30102f, 0.33811f, 0.37852f, 0.42280f, 0.46939f, 0.51795f,
     0.56926f, 0.62301f, 0.67933f, 0.73461f, 0.79041f, 0.84514f, 0.89604f, 0.94216f, 0.97906f, 1.0f},
    {0.00000f, 0.00196f, 0.00410f, 0.00652f, 0.00948f, 0.01288f, 0.01666f, 0.02092f, 0.02622f, 0.03215f, 0.03922f,
     0.04741f, 0.05764f, 0.06894f, 0.08259f, 0.09956f, 0.11886f, 0.14234f, 0.17014f, 0.20218f, 0.24143f, 0.28691f,
     0.33936f, 0.40088f, 0.46878f, 0.54607f, 0.62987f, 0.71605f, 0.80439f, 0.88842f, 0.95875f, 1.0f},
};

// sqrt(1 - albedo) of the half-space that reflects colour i / 31 in all: the table above inverted,
// linear in sqrt(1 - albedo) between its entries.
void BuildSubsurfaceRow(const float* reflectance, float* out)
{
    const auto root = [](u32 i) { return std::pow(1.0f - float(i) / float(kBsdfLutSize - 1), 1.5f); };
    for (u32 c = 0; c < kBsdfLutSize; ++c)
    {
        const float color = NodeValue(c);
        u32 i = 0;
        while (i + 2 < kBsdfLutSize && reflectance[i + 1] < color)
        {
            ++i;
        }
        const float span = reflectance[i + 1] - reflectance[i];
        const float t = span > 0.0f ? std::clamp((color - reflectance[i]) / span, 0.0f, 1.0f) : 0.0f;
        out[c] = root(i) + (root(i + 1) - root(i)) * t;
    }
}

std::vector<float> BuildLut()
{
    std::vector<float> lut(shader::kBsdfLutFloats, 0.0f);
    shader::tLut = lut.data();
    // Node 0 sits at grazing incidence; evaluate slightly above it to stay finite.
    const auto nodeMu = [](u32 i) { return std::max(NodeValue(i), 1e-3f); };
    for (u32 r = 0; r < kBsdfLutSize; ++r)
    {
        float* e = &lut[shader::kLutE + r * kBsdfLutSize];
        for (u32 m = 0; m < kBsdfLutSize; ++m)
        {
            e[m] = IntegrateSingleScatter(nodeMu(m), NodeValue(r), -1.0f);
        }
        lut[shader::kLutEavg + r] = CosineAverage(e);
    }
    // The dielectric coat: single scatter with F(0.04) plus its multiple-scattering lobe, whose
    // albedo is Fms * (1 - E(mu)) given the consistent Eavg above.
    for (u32 r = 0; r < kBsdfLutSize; ++r)
    {
        const float roughness = std::max(NodeValue(r), shader::kMinRoughness);
        const float fms = shader::MultiScatterFresnel(float3(shader::kDielectricF0), roughness).x;
        float* es = &lut[shader::kLutEs + r * kBsdfLutSize];
        const float* e = &lut[shader::kLutE + r * kBsdfLutSize];
        for (u32 m = 0; m < kBsdfLutSize; ++m)
        {
            es[m] = IntegrateSingleScatter(nodeMu(m), NodeValue(r), shader::kDielectricF0) + fms * (1.0f - e[m]);
        }
        lut[shader::kLutEsAvg + r] = CosineAverage(es);
    }
    for (u32 g = 0; g < shader::kSubsurfaceLutRows; ++g)
    {
        BuildSubsurfaceRow(kSubsurfaceReflectance[g], &lut[shader::kLutSubsurface + g * kBsdfLutSize]);
    }
    shader::tLut = nullptr;
    return lut;
}

const std::vector<float>& Lut()
{
    static const std::vector<float> lut = BuildLut();
    return lut;
}

void BindLut()
{
    shader::tLut = Lut().data();
}

static_assert(u32(CoatModel::Neural) == shader::kCoatNeural && u32(CoatModel::RandomWalk) == shader::kCoatRandomWalk &&
              u32(CoatModel::Approximate) == shader::kCoatApproximate);

shader::BsdfMaterial ToShader(const Material& m)
{
    shader::BsdfMaterial out;
    out.baseColor = float3(m.baseColor[0], m.baseColor[1], m.baseColor[2]);
    out.roughness = m.roughness;
    out.metallic = m.metallic;
    out.coat = shader::uint(m.coat);
    out.coatRoughness = m.coatRoughness;
    out.coatColor = float3(m.coatColor[0], m.coatColor[1], m.coatColor[2]);
    out.coatSeed = m.coatSeed;
    out.flakes = m.flakes;
    out.flakeRoughness = m.flakeRoughness;
    out.flakeColor = float3(m.flakeColor[0], m.flakeColor[1], m.flakeColor[2]);
    out.subsurfaceExit = m.subsurfaceExit ? 1u : 0u;
    return out;
}

float3 ToShader(const Direction& d)
{
    return float3(d[0], d[1], d[2]);
}

Rgb ToRgb(float3 v)
{
    return {v.x, v.y, v.z};
}

} // namespace

std::span<const float> GetBsdfLut()
{
    return Lut();
}

void BindNeuralCoat(std::span<const float> parameters)
{
    shader::tNeuralCoat = parameters.empty() ? nullptr : parameters.data();
}

BsdfValue EvaluateBsdf(const Material& material, const Direction& wo, const Direction& wi)
{
    BindLut();
    const shader::BsdfEval e = shader::EvaluateBsdf(ToShader(material), ToShader(wo), ToShader(wi));
    return {ToRgb(e.fCos), e.pdf};
}

BsdfDraw SampleBsdf(const Material& material, const Direction& wo, float uLobe, float u0, float u1)
{
    BindLut();
    const shader::BsdfSample s = shader::SampleBsdf(ToShader(material), ToShader(wo), uLobe, float2(u0, u1));
    return {{s.wi.x, s.wi.y, s.wi.z}, ToRgb(s.weight), s.pdf, s.valid};
}

Rgb EstimateAlbedo(const Material& material, float cosThetaO, u32 sampleCount)
{
    BindLut();
    shader::BsdfMaterial m = ToShader(material);
    const float3 wo(std::sqrt(std::max(0.0f, 1.0f - cosThetaO * cosThetaO)), 0.0f, cosThetaO);
    shader::uint seed = 0x9e3779b9u;
    double sum[3] = {};
    for (u32 i = 0; i < sampleCount; ++i)
    {
        m.coatSeed = shader::Pcg(material.coatSeed + i);
        const float uLobe = shader::Rand(seed);
        const float u0 = shader::Rand(seed);
        const float u1 = shader::Rand(seed);
        const shader::BsdfSample s = shader::SampleBsdf(m, wo, uLobe, float2(u0, u1));
        if (s.valid)
        {
            sum[0] += s.weight.x;
            sum[1] += s.weight.y;
            sum[2] += s.weight.z;
        }
    }
    return {float(sum[0] / sampleCount), float(sum[1] / sampleCount), float(sum[2] / sampleCount)};
}

Rgb EstimateAlbedoUniform(const Material& material, float cosThetaO, u32 sampleCount)
{
    BindLut();
    const shader::BsdfMaterial m = ToShader(material);
    const float3 wo(std::sqrt(std::max(0.0f, 1.0f - cosThetaO * cosThetaO)), 0.0f, cosThetaO);
    shader::uint seed = 0x85ebca6bu;
    double sum[3] = {};
    for (u32 i = 0; i < sampleCount; ++i)
    {
        const float z = shader::Rand(seed);
        const float phi = shader::kTwoPi * shader::Rand(seed);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float3 wi(r * std::cos(phi), r * std::sin(phi), z);
        const float3 f = shader::EvaluateBsdf(m, wo, wi).fCos;
        sum[0] += f.x;
        sum[1] += f.y;
        sum[2] += f.z;
    }
    const double scale = shader::kTwoPi / double(sampleCount); // pdf = 1 / (2 pi)
    return {float(sum[0] * scale), float(sum[1] * scale), float(sum[2] * scale)};
}

BsdfValue EvaluateSurfaceBsdf(const Material& material, const Direction& wo, const Direction& ng, const Direction& wi)
{
    BindLut();
    const shader::BsdfEval e =
        shader::EvaluateSurfaceBsdf(ToShader(material), ToShader(wo), ToShader(ng), ToShader(wi));
    return {ToRgb(e.fCos), e.pdf};
}

BsdfDraw SampleSurfaceBsdf(const Material& material, const Direction& wo, const Direction& ng, float uLobe, float u0,
                           float u1)
{
    BindLut();
    const shader::BsdfSample s =
        shader::SampleSurfaceBsdf(ToShader(material), ToShader(wo), ToShader(ng), uLobe, float2(u0, u1));
    return {{s.wi.x, s.wi.y, s.wi.z}, ToRgb(s.weight), s.pdf, s.valid};
}

Rgb EstimateSurfaceAlbedo(const Material& material, const Direction& wo, const Direction& ng, u32 sampleCount)
{
    BindLut();
    const shader::BsdfContext c = shader::PrepareBsdf(ToShader(material), ToShader(wo));
    const float3 n = ToShader(ng);
    shader::uint seed = 0x27d4eb2fu;
    double sum[3] = {};
    for (u32 i = 0; i < sampleCount; ++i)
    {
        const float uLobe = shader::Rand(seed);
        const float u0 = shader::Rand(seed);
        const float u1 = shader::Rand(seed);
        const shader::BsdfSample s = shader::SampleSurfaceBsdf(c, n, uLobe, float2(u0, u1));
        if (s.valid)
        {
            sum[0] += s.weight.x;
            sum[1] += s.weight.y;
            sum[2] += s.weight.z;
        }
    }
    return {float(sum[0] / sampleCount), float(sum[1] / sampleCount), float(sum[2] / sampleCount)};
}

Rgb EstimateSurfaceAlbedoUniform(const Material& material, const Direction& wo, const Direction& ng, u32 sampleCount)
{
    BindLut();
    const shader::BsdfContext c = shader::PrepareBsdf(ToShader(material), ToShader(wo));
    const float3 n = ToShader(ng);
    const shader::Frame frame = shader::MakeFrame(n);
    shader::uint seed = 0x165667b1u;
    double sum[3] = {};
    for (u32 i = 0; i < sampleCount; ++i)
    {
        const float z = shader::Rand(seed);
        const float phi = shader::kTwoPi * shader::Rand(seed);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float3 wi = shader::ToWorld(frame, float3(r * std::cos(phi), r * std::sin(phi), z));
        const float3 f = shader::EvaluateSurfaceBsdf(c, n, wi).fCos;
        sum[0] += f.x;
        sum[1] += f.y;
        sum[2] += f.z;
    }
    const double scale = shader::kTwoPi / double(sampleCount); // pdf = 1 / (2 pi) around ng
    return {float(sum[0] * scale), float(sum[1] * scale), float(sum[2] * scale)};
}

} // namespace air::pt
