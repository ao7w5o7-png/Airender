// Compiles shaders/Bsdf.hlsli as C++ and builds its lookup tables.

#include "air/features/pathtracer/Bsdf.h"

#include "air/renderer/shader/HlslCompat.h"

#include <vector>

namespace air::pt {
namespace shader {

using namespace air::hlsl;

// The table BsdfLutLoad() reads. Points at the finished LUT, or at the partial one while it is built.
thread_local const float* tLut = nullptr;

float BsdfLutLoad(uint index);

#include "Bsdf.hlsli"

float BsdfLutLoad(uint index) {
    return tLut[index];
}

} // namespace shader

namespace {

using shader::float2;
using shader::float3;
using shader::kBsdfLutSize;

// Stratified sample grid per table entry: kGrid^2 visible normal samples.
constexpr u32 kGrid = 48;

float NodeValue(u32 i) {
    return float(i) / float(kBsdfLutSize - 1);
}

// Directional albedo of single-scatter GGX, optionally with Schlick Fresnel (f0 >= 0).
float IntegrateSingleScatter(float mu, float roughness, float f0) {
    const float alpha = shader::RoughnessToAlpha(roughness);
    const float3 wo(std::sqrt(1.0f - mu * mu), 0.0f, mu);
    const float g1o = shader::GgxG1(wo, alpha);
    double sum = 0.0;
    for (u32 y = 0; y < kGrid; ++y) {
        for (u32 x = 0; x < kGrid; ++x) {
            const float2 u((float(x) + 0.5f) / kGrid, (float(y) + 0.5f) / kGrid);
            const float3 h = shader::SampleGgxVndf(wo, alpha, u);
            const float3 wi = shader::reflect(-wo, h);
            if (wi.z <= 0.0f) {
                continue;
            }
            // f cos / pdf = G2 / G1(wo) for visible normal sampling.
            float weight = shader::GgxG2(wo, wi, alpha) / g1o;
            if (f0 >= 0.0f) {
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
float CosineAverage(const float* row) {
    double sum = 0.0;
    for (u32 i = 0; i + 1 < kBsdfLutSize; ++i) {
        const double a = NodeValue(i);
        const double b = NodeValue(i + 1);
        sum += (b - a) / 6.0 * (row[i] * (2.0 * a + b) + row[i + 1] * (a + 2.0 * b));
    }
    return float(2.0 * sum);
}

std::vector<float> BuildLut() {
    std::vector<float> lut(shader::kBsdfLutFloats, 0.0f);
    shader::tLut = lut.data();
    // Node 0 sits at grazing incidence; evaluate slightly above it to stay finite.
    const auto nodeMu = [](u32 i) { return std::max(NodeValue(i), 1e-3f); };
    for (u32 r = 0; r < kBsdfLutSize; ++r) {
        float* e = &lut[shader::kLutE + r * kBsdfLutSize];
        for (u32 m = 0; m < kBsdfLutSize; ++m) {
            e[m] = IntegrateSingleScatter(nodeMu(m), NodeValue(r), -1.0f);
        }
        lut[shader::kLutEavg + r] = CosineAverage(e);
    }
    // The dielectric coat: single scatter with F(0.04) plus its multiple-scattering lobe, whose
    // albedo is Fms * (1 - E(mu)) given the consistent Eavg above.
    for (u32 r = 0; r < kBsdfLutSize; ++r) {
        const float roughness = std::max(NodeValue(r), shader::kMinRoughness);
        const float fms = shader::MultiScatterFresnel(float3(shader::kDielectricF0), roughness).x;
        float* es = &lut[shader::kLutEs + r * kBsdfLutSize];
        const float* e = &lut[shader::kLutE + r * kBsdfLutSize];
        for (u32 m = 0; m < kBsdfLutSize; ++m) {
            es[m] = IntegrateSingleScatter(nodeMu(m), NodeValue(r), shader::kDielectricF0) + fms * (1.0f - e[m]);
        }
        lut[shader::kLutEsAvg + r] = CosineAverage(es);
    }
    shader::tLut = nullptr;
    return lut;
}

const std::vector<float>& Lut() {
    static const std::vector<float> lut = BuildLut();
    return lut;
}

void BindLut() {
    shader::tLut = Lut().data();
}

shader::BsdfMaterial ToShader(const Material& m) {
    shader::BsdfMaterial out;
    out.baseColor = float3(m.baseColor[0], m.baseColor[1], m.baseColor[2]);
    out.roughness = m.roughness;
    out.metallic = m.metallic;
    return out;
}

float3 ToShader(const Direction& d) {
    return float3(d[0], d[1], d[2]);
}

Rgb ToRgb(float3 v) {
    return {v.x, v.y, v.z};
}

} // namespace

std::span<const float> GetBsdfLut() {
    return Lut();
}

BsdfValue EvaluateBsdf(const Material& material, const Direction& wo, const Direction& wi) {
    BindLut();
    const shader::BsdfEval e = shader::EvaluateBsdf(ToShader(material), ToShader(wo), ToShader(wi));
    return {ToRgb(e.fCos), e.pdf};
}

BsdfDraw SampleBsdf(const Material& material, const Direction& wo, float uLobe, float u0, float u1) {
    BindLut();
    const shader::BsdfSample s = shader::SampleBsdf(ToShader(material), ToShader(wo), uLobe, float2(u0, u1));
    return {{s.wi.x, s.wi.y, s.wi.z}, ToRgb(s.weight), s.pdf, s.valid};
}

Rgb EstimateAlbedo(const Material& material, float cosThetaO, u32 sampleCount) {
    BindLut();
    const shader::BsdfMaterial m = ToShader(material);
    const float3 wo(std::sqrt(std::max(0.0f, 1.0f - cosThetaO * cosThetaO)), 0.0f, cosThetaO);
    uint seed = 0x9e3779b9u;
    double sum[3] = {};
    for (u32 i = 0; i < sampleCount; ++i) {
        const float uLobe = shader::Rand(seed);
        const float u0 = shader::Rand(seed);
        const float u1 = shader::Rand(seed);
        const shader::BsdfSample s = shader::SampleBsdf(m, wo, uLobe, float2(u0, u1));
        if (s.valid) {
            sum[0] += s.weight.x;
            sum[1] += s.weight.y;
            sum[2] += s.weight.z;
        }
    }
    return {float(sum[0] / sampleCount), float(sum[1] / sampleCount), float(sum[2] / sampleCount)};
}

Rgb EstimateAlbedoUniform(const Material& material, float cosThetaO, u32 sampleCount) {
    BindLut();
    const shader::BsdfMaterial m = ToShader(material);
    const float3 wo(std::sqrt(std::max(0.0f, 1.0f - cosThetaO * cosThetaO)), 0.0f, cosThetaO);
    uint seed = 0x85ebca6bu;
    double sum[3] = {};
    for (u32 i = 0; i < sampleCount; ++i) {
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

} // namespace air::pt
