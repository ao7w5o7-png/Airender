// Energy conservation and consistency of the path tracer BSDF, run on the shader source itself.

#include "TestFramework.h"

#include "air/features/pathtracer/Bsdf.h"

#include <cmath>
#include <cstdio>

using air::pt::Material;
using air::pt::Rgb;

namespace {

constexpr float kRoughness[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f};
constexpr float kCosTheta[] = {0.05f, 0.2f, 0.5f, 0.8f, 1.0f};

bool Near(float value, float expected, float tolerance, const char* what, const Material& m, float mu) {
    if (std::abs(value - expected) <= tolerance) {
        return true;
    }
    std::fprintf(stderr, "  %s = %.4f, expected %.4f +- %.4f (roughness %.2f metallic %.2f mu %.2f)\n", what, value,
                 expected, tolerance, m.roughness, m.metallic, mu);
    return false;
}

} // namespace

// A white material must reflect all incoming energy at every angle and roughness: the furnace
// test fails when energy is lost (missing multiple scattering) or created (double counting).
TEST(Bsdf_WhiteFurnaceMetal) {
    for (float roughness : kRoughness) {
        for (float mu : kCosTheta) {
            const Material m{{1.0f, 1.0f, 1.0f}, roughness, 1.0f};
            const Rgb albedo = air::pt::EstimateAlbedo(m, mu, 1 << 15);
            EXPECT(Near(albedo[0], 1.0f, 0.02f, "metal albedo", m, mu));
        }
    }
}

TEST(Bsdf_WhiteFurnaceDielectric) {
    for (float roughness : kRoughness) {
        for (float mu : kCosTheta) {
            const Material m{{1.0f, 1.0f, 1.0f}, roughness, 0.0f};
            const Rgb albedo = air::pt::EstimateAlbedo(m, mu, 1 << 15);
            EXPECT(Near(albedo[1], 1.0f, 0.02f, "dielectric albedo", m, mu));
        }
    }
}

TEST(Bsdf_NeverReflectsMoreThanItReceives) {
    for (float roughness : kRoughness) {
        for (float metallic : {0.0f, 0.5f, 1.0f}) {
            const Material m{{0.9f, 0.5f, 0.1f}, roughness, metallic};
            const Rgb albedo = air::pt::EstimateAlbedo(m, 0.6f, 1 << 14);
            for (float channel : albedo) {
                EXPECT(channel >= 0.0f && channel <= 1.01f);
            }
        }
    }
}

// The sampler's weights must integrate to the same albedo as brute-force uniform sampling.
TEST(Bsdf_SamplingMatchesUniformIntegration) {
    for (float roughness : {0.4f, 0.7f, 1.0f}) {
        for (float metallic : {0.0f, 1.0f}) {
            const Material m{{0.7f, 0.4f, 0.2f}, roughness, metallic};
            const Rgb sampled = air::pt::EstimateAlbedo(m, 0.7f, 1 << 16);
            const Rgb uniform = air::pt::EstimateAlbedoUniform(m, 0.7f, 1 << 18);
            for (int c = 0; c < 3; ++c) {
                EXPECT(Near(sampled[c], uniform[c], 0.015f, "sampled vs uniform", m, 0.7f));
            }
        }
    }
}

TEST(Bsdf_Reciprocity) {
    const air::pt::Direction a = {0.3f, -0.2f, 0.932738f};
    const air::pt::Direction b = {-0.6f, 0.5f, 0.6244998f};
    for (float roughness : kRoughness) {
        for (float metallic : {0.0f, 0.5f, 1.0f}) {
            const Material m{{0.8f, 0.6f, 0.3f}, roughness, metallic};
            const air::pt::BsdfValue ab = air::pt::EvaluateBsdf(m, a, b);
            const air::pt::BsdfValue ba = air::pt::EvaluateBsdf(m, b, a);
            for (int c = 0; c < 3; ++c) {
                const float fab = ab.fCos[c] / b[2];
                const float fba = ba.fCos[c] / a[2];
                EXPECT(std::abs(fab - fba) <= 1e-4f * std::max(1.0f, fab));
            }
        }
    }
}

TEST(Bsdf_SampleWeightMatchesEvaluation) {
    const Material m{{0.5f, 0.6f, 0.7f}, 0.35f, 0.3f};
    const air::pt::Direction wo = {0.6f, 0.0f, 0.8f};
    const air::pt::BsdfDraw s = air::pt::SampleBsdf(m, wo, 0.3f, 0.42f, 0.77f);
    EXPECT(s.valid);
    const air::pt::BsdfValue e = air::pt::EvaluateBsdf(m, wo, s.wi);
    EXPECT(std::abs(e.pdf - s.pdf) <= 1e-5f * std::max(1.0f, e.pdf));
    EXPECT(std::abs(e.fCos[0] / e.pdf - s.weight[0]) <= 1e-4f);
}

TEST(Bsdf_LutIsBoundedAndMonotonic) {
    const auto lut = air::pt::GetBsdfLut();
    for (float v : lut) {
        EXPECT(v >= 0.0f && v <= 1.0f);
    }
    // At normal incidence a smooth mirror reflects everything and roughness only loses energy.
    EXPECT(lut[31] > 0.99f);
    EXPECT(lut[31 * 32 + 31] < lut[31]);
}
