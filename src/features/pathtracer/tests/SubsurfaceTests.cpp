// Subsurface scattering (Subsurface.hlsli, SubsurfaceFit.cpp, docs/architecture.md §8.20): the medium's
// half-space reflects the material's colour in all, the neural profile follows the walk closer than
// Burley's, and the integrator neither creates nor loses light with any of the three models.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/features/pathtracer/ReferenceRenderer.h"
#include "air/features/pathtracer/SubsurfaceFit.h"

#include <cmath>
#include <cstdio>

using namespace air;
using renderer::float3;

namespace
{

double MeanLuminance(const Image& image, u32 x0, u32 y0, u32 x1, u32 y1)
{
    double sum = 0.0;
    for (u32 y = y0; y < y1; ++y)
    {
        for (u32 x = x0; x < x1; ++x)
        {
            const float* p = image.At(x, y);
            sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        }
    }
    return sum / (double(x1 - x0) * double(y1 - y0));
}

double MeanLuminance(const Image& image)
{
    return MeanLuminance(image, 0, 0, image.width, image.height);
}

bool Near(double value, double expected, double tolerance, const char* what)
{
    if (std::abs(value - expected) <= tolerance)
    {
        return true;
    }
    std::fprintf(stderr, "  %s = %.5f, expected %.5f +- %.5f\n", what, value, expected, tolerance);
    return false;
}

pt::ReferenceSettings Settings(TaskSystem& tasks, u32 spp, u32 model)
{
    pt::ReferenceSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.samplesPerPixel = spp;
    settings.maxBounces = 16;
    settings.subsurfaceModel = model;
    settings.tasks = &tasks;
    return settings;
}

} // namespace

// The half-space of the medium the BSDF LUT's inversion chooses reflects the colour in all.
TEST(Subsurface_HalfSpaceReflectsItsColour)
{
    TaskSystem tasks;
    const pt::SubsurfaceProfileTable::Config configs[] = {{0.2f, 0.0f}, {0.5f, 0.3f},  {0.8f, 0.0f},
                                                          {0.8f, 0.6f}, {0.95f, 0.9f}, {0.7f, 0.45f}};
    const pt::SubsurfaceProfileTable table = pt::SimulateSubsurfaceProfiles(configs, 40000, 7, &tasks);
    for (size_t c = 0; c < table.configs.size(); ++c)
    {
        const double a = table.configs[c].color;
        EXPECT(Near(table.total[c], a, 0.01 + 4.0 * std::sqrt(a * (1.0 - a) / 40000.0), "half-space reflectance"));
    }
}

// The network the shaders compile is the one the CPU evaluates.
TEST(Subsurface_ShaderNetworkMatchesTheCpu)
{
    const std::span<const float> p = pt::CompiledSubsurfaceProfile();
    for (float logRadius : {-5.0f, -2.0f, 0.0f, 1.5f, 3.0f})
    {
        for (float color : {0.1f, 0.5f, 0.9f})
        {
            for (float g : {0.0f, 0.45f, 0.9f})
            {
                EXPECT(std::abs(pt::ShaderSubsurfaceProfileCorrection(logRadius, color, g) -
                                pt::EvaluateSubsurfaceProfileCorrection(p, logRadius, color, g)) < 1e-5f);
            }
        }
    }
}

// The compiled network follows the walk's profile far closer than Burley's, and keeps its energy.
TEST(Subsurface_NeuralProfileFollowsTheWalk)
{
    TaskSystem tasks;
    const pt::SubsurfaceProfileTable::Config configs[] = {{0.3f, 0.0f}, {0.6f, 0.15f}, {0.85f, 0.3f},
                                                          {0.9f, 0.6f}, {0.5f, 0.75f}, {0.97f, 0.0f}};
    const pt::SubsurfaceProfileTable table = pt::SimulateSubsurfaceProfiles(configs, 60000, 11, &tasks);
    json::Value report = pt::CheckSubsurfaceProfile(table, pt::CompiledSubsurfaceProfile());
    const double burley = report["burleyRms"].AsNumber();
    const double network = report["networkRms"].AsNumber();
    std::printf("  profile error: Burley %.3f, network %.3f; energy error: Burley %.4f, network %.4f\n", burley,
                network, report["burleyEnergyError"].AsNumber(), report["networkEnergyError"].AsNumber());
    EXPECT(network < 0.5 * burley);
    EXPECT(report["networkEnergyError"].AsNumber() < 0.03);
}

// White spheres of a subsurface material vanish into a uniform environment: the light that goes
// beneath leaves again, all of it, with every model.
TEST(Subsurface_WhiteFurnace)
{
    TaskSystem tasks;
    renderer::SceneDesc scene = renderer::MakeFurnaceScene();
    for (renderer::Mesh& mesh : scene.meshes)
    {
        mesh = renderer::MakeSphere(mesh.name, 0.2f, 24);
    }
    for (renderer::Material& m : scene.materials)
    {
        m.metallic = 0.0f;
        m.coat = false;
        m.subsurfaceRadius = {0.05f, 0.02f, 0.01f};
    }
    for (u32 model : {0u, 1u, 2u})
    {
        pt::ReferenceSettings settings = Settings(tasks, model == 1u ? 64 : 32, model);
        settings.width = 48;
        settings.furnace = true;
        const double mean = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
        std::printf("  model %u: furnace mean %.4f\n", model, mean);
        // The walk is exact; the profiles lose a little on the spheres' curvature.
        EXPECT(Near(mean, 0.18, model == 1u ? 0.002 : 0.005, "subsurface furnace mean"));
    }
}

// A scene with subsurface materials runs the kernels that carry its light beneath (the neural variants)
// with every model; materials out of range do not validate.
TEST(Subsurface_ScenesRunTheirKernels)
{
    airtest::Headless h({"pathtracer", "tonemap"});
    h.device->WaitIdle();
    h.scene = std::make_unique<renderer::Scene>(*h.device, airtest::BuiltinScene("subsurface"));
    h.camera = h.scene->Desc().camera;
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (const char* model : {"0", "1", "2"})
    {
        EXPECT(h.runner->SetParameter("pathtracer.subsurfaceModel", model));
        const renderer::FrameResult frame = h.Frame();
        EXPECT(frame.validationErrors.empty());
        EXPECT(airtest::ContainsLine(frame.commands, "SetPipeline pipeline 'PathTracer.TraceGiNeural'"));
    }
    EXPECT(!h.runner->SetParameter("pathtracer.subsurfaceModel", "3"));

    renderer::SceneDesc bad = airtest::BuiltinScene("subsurface");
    bad.materials[0].metallic = 0.5f;
    EXPECT(!renderer::Validate(bad).empty());
    bad = airtest::BuiltinScene("subsurface");
    bad.materials[0].subsurfaceRadius = {-0.01f, 0.0f, 0.0f};
    EXPECT(!renderer::Validate(bad).empty());
}

// A thick slab lit by a uniform sky: far from its edges subsurface scattering moves the light around
// without changing how much the surface reflects, as the plain dielectric of its colour.
TEST(Subsurface_FlatSlabReflectsAsTheDiffuseLimit)
{
    TaskSystem tasks;
    renderer::SceneDesc scene;
    scene.name = "slab";
    scene.environment = {1.0f, 1.0f, 1.0f};
    scene.meshes.push_back(renderer::MakeBox("Slab", {4.0f, 1.0f, 4.0f}));
    scene.materials.push_back({.name = "Plain", .baseColor = {0.7f, 0.5f, 0.3f}, .roughness = 0.4f});
    renderer::Instance slab;
    slab.name = "Slab";
    scene.instances.push_back(slab);
    // Looking straight down at the slab's middle from 2 m above it.
    scene.camera.position = {0.0f, 3.0f, 0.0f};
    scene.camera.pitch = -1.5707f;
    scene.camera.verticalFov = 0.3f;
    pt::ReferenceSettings settings = Settings(tasks, 64, 1);
    settings.applyExposure = false;
    const double plain = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    scene.materials[0].subsurfaceRadius = {0.05f, 0.03f, 0.02f};
    for (u32 model : {0u, 1u, 2u})
    {
        settings.subsurfaceModel = model;
        const double beneath = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
        std::printf("  model %u: %.4f, plain %.4f\n", model, beneath, plain);
        EXPECT(Near(beneath, plain, 0.015 * plain, "slab's reflectance"));
    }
}
