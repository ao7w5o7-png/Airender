// Clear coats (Bsdf.hlsli, docs/architecture.md §8.17): the random walk that is their reference, the
// sampling that serves all three models, and the neural coat trained from the walk; on the shader
// source itself.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/features/pathtracer/Bsdf.h"
#include "air/features/pathtracer/NeuralCoat.h"

#include <cmath>
#include <cstdio>

using air::pt::CoatModel;
using air::pt::Material;
using air::pt::Rgb;

namespace
{

Material Coated(Rgb base, float roughness, float metallic, float coatRoughness, Rgb coatColor = {1.0f, 1.0f, 1.0f},
                CoatModel model = CoatModel::RandomWalk)
{
    Material m;
    m.baseColor = base;
    m.roughness = roughness;
    m.metallic = metallic;
    m.coat = model;
    m.coatRoughness = coatRoughness;
    m.coatColor = coatColor;
    return m;
}

air::pt::Direction FromCos(float mu, float phi = 0.0f)
{
    const float s = std::sqrt(std::max(0.0f, 1.0f - mu * mu));
    return {s * std::cos(phi), s * std::sin(phi), mu};
}

} // namespace

// A coat over a white base that absorbs nothing reflects all the light (the interface's single
// scattering loses a little at grazing angles when rough): the walk neither loses nor creates energy
// on its way between the interface and the base.
TEST(Coat_WhiteFurnaceRandomWalk)
{
    float lowest = 2.0f;
    float highest = 0.0f;
    for (float coatRoughness : {0.03f, 0.3f})
    {
        for (float metallic : {0.0f, 1.0f})
        {
            for (float mu : {0.2f, 0.5f, 0.9f})
            {
                const Material m = Coated({1.0f, 1.0f, 1.0f}, 0.5f, metallic, coatRoughness);
                const float albedo = air::pt::EstimateAlbedo(m, mu, 1 << 15)[1];
                lowest = std::min(lowest, albedo);
                highest = std::max(highest, albedo);
                const bool ok = albedo > 0.94f && albedo < 1.02f;
                if (!ok)
                {
                    std::fprintf(stderr, "  albedo %.4f (coat roughness %.2f metallic %.0f mu %.1f)\n", albedo,
                                 coatRoughness, metallic, mu);
                }
                EXPECT(ok);
            }
        }
    }
    std::printf("  albedo %.3f .. %.3f\n", lowest, highest);
}

Material Flaked(Rgb pigment, Rgb flakeColor, float flakes, float flakeRoughness,
                CoatModel model = CoatModel::RandomWalk)
{
    Material m = Coated(pigment, 0.8f, 0.0f, 0.03f, {1.0f, 1.0f, 1.0f}, model);
    m.flakes = flakes;
    m.flakeRoughness = flakeRoughness;
    m.flakeColor = flakeColor;
    return m;
}

// Car paint (Material::flakes): white flakes in a white pigment under a clear coat reflect all the
// light too, whatever their share and spread; coloured flakes keep their colour where they reflect
// (towards the mirror direction) and the pigment's elsewhere.
TEST(Coat_FlakesConserveEnergy)
{
    for (float flakes : {0.3f, 0.8f})
    {
        for (float flakeRoughness : {0.2f, 0.6f})
        {
            for (float mu : {0.3f, 0.9f})
            {
                const Material m = Flaked({1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, flakes, flakeRoughness);
                const float albedo = air::pt::EstimateAlbedo(m, mu, 1 << 15)[1];
                const bool ok = albedo > 0.94f && albedo < 1.02f;
                if (!ok)
                {
                    std::fprintf(stderr, "  albedo %.4f (flakes %.1f roughness %.1f mu %.1f)\n", albedo, flakes,
                                 flakeRoughness, mu);
                }
                EXPECT(ok);
            }
        }
    }
    // Gold flakes in a blue pigment: towards the mirror direction the flakes' red dominates.
    const Material paint = Flaked({0.05f, 0.1f, 0.6f}, {1.0f, 0.75f, 0.3f}, 0.5f, 0.15f);
    const air::pt::Direction wo = FromCos(0.7f);
    const air::pt::Direction mirror = {-wo[0], -wo[1], wo[2]};
    const air::pt::Direction away = FromCos(0.9f, 1.4f);
    Rgb atMirror{};
    Rgb atAway{};
    for (air::u32 i = 0; i < 1 << 12; ++i)
    {
        Material m = paint;
        m.coatSeed = i * 2654435761u;
        const auto a = air::pt::EvaluateBsdf(m, wo, mirror);
        const auto b = air::pt::EvaluateBsdf(m, wo, away);
        for (int c = 0; c < 3; ++c)
        {
            atMirror[c] += a.fCos[c];
            atAway[c] += b.fCos[c];
        }
    }
    EXPECT(atMirror[0] > atMirror[2]);
    EXPECT(atAway[2] > atAway[0]);
}

// The layer's colour absorbs on the way in and out: twice through a layer of 0.5 at normal incidence
// leaves well under half of the inner light, the interface's reflection unchanged.
TEST(Coat_AbsorptionDarkensTheInnerLight)
{
    const Material clear = Coated({0.8f, 0.8f, 0.8f}, 0.6f, 0.0f, 0.05f);
    const Material tinted = Coated({0.8f, 0.8f, 0.8f}, 0.6f, 0.0f, 0.05f, {0.5f, 0.5f, 0.5f});
    const float a = air::pt::EstimateAlbedo(clear, 0.9f, 1 << 15)[0];
    const float b = air::pt::EstimateAlbedo(tinted, 0.9f, 1 << 15)[0];
    EXPECT(b < 0.4f * a + 0.05f && b > 0.04f);
}

// f(wo, wi) = f(wi, wo): the walk's estimates, averaged, agree both ways round.
TEST(Coat_RandomWalkIsReciprocal)
{
    for (float metallic : {0.0f, 1.0f})
    {
        Material m = Coated({0.7f, 0.5f, 0.3f}, 0.4f, metallic, 0.1f, {0.9f, 0.8f, 0.7f});
        const air::pt::Direction a = FromCos(0.8f);
        const air::pt::Direction b = FromCos(0.4f, 2.5f);
        double ab = 0.0;
        double ba = 0.0;
        for (air::u32 i = 0; i < 1 << 14; ++i)
        {
            m.coatSeed = i * 2654435761u;
            ab += air::pt::EvaluateBsdf(m, a, b).fCos[0] / b[2];
            ba += air::pt::EvaluateBsdf(m, b, a).fCos[0] / a[2];
        }
        const double ratio = ab / ba;
        if (std::abs(ratio - 1.0) > 0.03)
        {
            std::fprintf(stderr, "  f(a, b) / f(b, a) = %.4f (metallic %.0f)\n", ratio, metallic);
        }
        EXPECT(std::abs(ratio - 1.0) <= 0.03);
    }
}

// Sampling (the coat's reflection or the base's lobes) and evaluation agree: the albedo by the
// sampler's weights equals the integral of the evaluations over the hemisphere, for every model.
TEST(Coat_SamplingMatchesUniformIntegration)
{
    for (CoatModel model : {CoatModel::RandomWalk, CoatModel::Approximate})
    {
        for (float mu : {0.3f, 0.8f})
        {
            Material m = Coated({0.6f, 0.4f, 0.2f}, 0.5f, 0.3f, 0.2f, {0.9f, 0.9f, 0.8f}, model);
            // Flakes too: their lobe is part of the sampling.
            m.flakes = mu > 0.5f ? 0.4f : 0.0f;
            m.flakeRoughness = 0.25f;
            const float sampled = air::pt::EstimateAlbedo(m, mu, 1 << 16)[0];
            const float uniform = air::pt::EstimateAlbedoUniform(m, mu, 1 << 17)[0];
            if (std::abs(sampled - uniform) > 0.02f)
            {
                std::fprintf(stderr, "  model %u mu %.1f: sampled %.4f, uniform %.4f\n", air::u32(model), mu, sampled,
                             uniform);
            }
            EXPECT(std::abs(sampled - uniform) <= 0.02f);
        }
    }
}

// The network learns the walk: for the coated scene's materials, after a short training, its error
// against many walks is a fraction of the approximation's, and the training is found again in the cache.
TEST(NeuralCoat_LearnsTheRandomWalk)
{
    air::TaskSystem tasks;
    const auto materials = air::pt::CollectCoatMaterials(airtest::BuiltinScene("coated"), 256);
    EXPECT(materials.size() >= 4);
    air::pt::NeuralCoatSettings settings;
    settings.steps = 300;
    settings.batch = 2048;
    settings.cache = std::filesystem::temp_directory_path() / "airender-coat-tests";
    std::error_code ignored;
    std::filesystem::remove_all(settings.cache, ignored);
    const air::pt::NeuralCoatResult trained = air::pt::TrainNeuralCoat(materials, settings, &tasks);
    EXPECT(!trained.cached && trained.parameters.size() == 980);
    air::json::Value check = air::pt::CheckNeuralCoat(materials, trained.parameters, 64, &tasks);
    const double neural = check["neuralError"].AsNumber();
    const double approximate = check["approximateError"].AsNumber();
    std::printf("  neural %.3f, approximation %.3f\n", neural, approximate);
    EXPECT(neural < 0.5 * approximate && neural < 0.08);
    const air::pt::NeuralCoatResult again = air::pt::TrainNeuralCoat(materials, settings, &tasks);
    EXPECT(again.cached && again.parameters == trained.parameters);
    std::filesystem::remove_all(settings.cache, ignored);
}

// A scene with coated materials runs the trace's and ReSTIR's kernels with the coats (the neural
// variants) and reports the network's training; a scene without runs the plain ones. Coat values out
// of range do not validate.
TEST(Coat_ScenesRunTheirKernels)
{
    airtest::Headless h({"pathtracer", "tonemap"});
    h.device->WaitIdle();
    h.scene = std::make_unique<air::renderer::Scene>(*h.device, airtest::BuiltinScene("coated"));
    h.camera = h.scene->Desc().camera;
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    air::renderer::FrameResult frame = h.Frame();
    EXPECT(frame.validationErrors.empty());
    EXPECT(airtest::ContainsLine(frame.commands, "SetPipeline pipeline 'PathTracer.TraceGiNeural'"));
    EXPECT(airtest::ContainsLine(frame.commands, "SetPipeline pipeline 'PathTracer.ReSTIRGI.SpatialNeural'"));
    EXPECT(h.runner->DescribeFeatures().Dump(0).find("\"neuralCoatLoss\":0,") == std::string::npos);
    for (const char* model : {"1", "2"})
    {
        EXPECT(h.runner->SetParameter("pathtracer.coatModel", model));
        EXPECT(h.Frame().validationErrors.empty());
    }
    EXPECT(!h.runner->SetParameter("pathtracer.coatModel", "3"));
    h.device->WaitIdle();
    h.scene = std::make_unique<air::renderer::Scene>(*h.device, airtest::BuiltinScene("textured"));
    frame = h.Frame();
    EXPECT(!airtest::ContainsLine(frame.commands, "Neural'"));

    air::renderer::SceneDesc bad = airtest::BuiltinScene("coated");
    bad.materials[0].coat = true;
    bad.materials[0].coatColor = {1.5f, 1.0f, 1.0f};
    EXPECT(!air::renderer::Validate(bad).empty());
}
