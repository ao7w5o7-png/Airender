// The path tracer's estimator (Integrator.hlsli) run on the CPU against analytic answers.

#include "TestFramework.h"

#include "air/features/pathtracer/ReferenceRenderer.h"

#include <cmath>
#include <cstdio>

using namespace air;
using renderer::float3;

namespace
{

double MeanLuminance(const Image& image)
{
    double sum = 0.0;
    for (u32 y = 0; y < image.height; ++y)
    {
        for (u32 x = 0; x < image.width; ++x)
        {
            const float* p = image.At(x, y);
            sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        }
    }
    return sum / (double(image.width) * image.height);
}

bool Near(double value, double expected, double relative, const char* what)
{
    if (std::abs(value - expected) <= relative * std::abs(expected))
    {
        return true;
    }
    std::fprintf(stderr, "  %s = %.4f, expected %.4f +- %.1f%%\n", what, value, expected, relative * 100.0);
    return false;
}

// A closed 2 m cube seen from its centre, every wall facing inwards with `wall` and, optionally,
// `ceiling` on the top face.
renderer::SceneDesc Room(const renderer::Material& wall, const renderer::Material* ceiling)
{
    renderer::SceneDesc scene;
    scene.name = "room";
    scene.materials.push_back(wall);
    scene.materials.push_back(ceiling ? *ceiling : wall);
    const float3 c[8] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                         {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    struct Face
    {
        int a, b, d, e;
        float3 inward;
    };
    const Face faces[6] = {{0, 1, 2, 3, {0, 0, 1}},  {4, 5, 6, 7, {0, 0, -1}}, {0, 3, 7, 4, {1, 0, 0}},
                           {1, 2, 6, 5, {-1, 0, 0}}, {0, 1, 5, 4, {0, 1, 0}},  {3, 2, 6, 7, {0, -1, 0}}};
    for (int i = 0; i < 6; ++i)
    {
        const Face& f = faces[i];
        scene.meshes.push_back(renderer::MakeQuad("Wall", c[f.a], c[f.b], c[f.d], c[f.e], f.inward));
        renderer::Instance instance;
        instance.name = "Wall";
        instance.mesh = u32(i);
        instance.material = i == 5 ? 1u : 0u; // face 5 is the ceiling (y = +1)
        scene.instances.push_back(instance);
    }
    scene.camera.position = {0.0f, 0.0f, 0.0f};
    scene.camera.verticalFov = 1.5f;
    return scene;
}

pt::ReferenceSettings Settings(TaskSystem& tasks, u32 spp, u32 bounces)
{
    pt::ReferenceSettings settings;
    settings.width = 24;
    settings.height = 24;
    settings.samplesPerPixel = spp;
    settings.maxBounces = bounces;
    settings.applyExposure = false;
    settings.tasks = &tasks;
    return settings;
}

} // namespace

// Every wall emits 1 nit and reflects all light (white, energy conserving), so a path that
// scatters B times collects exactly B + 1 nits: the image must be flat B + 1 with or without
// NEE. Catches any MIS weight, light pdf or BSDF energy error.
TEST(Integrator_EmissiveWhiteRoomCollectsOneNitPerBounce)
{
    TaskSystem tasks(4);
    const u32 bounces = 4;
    for (float metallic : {0.0f, 1.0f})
    {
        for (float roughness : {0.2f, 0.8f})
        {
            const renderer::Material white{.name = "White",
                                           .baseColor = {1.0f, 1.0f, 1.0f},
                                           .roughness = roughness,
                                           .metallic = metallic,
                                           .emissionNits = 1.0f};
            const renderer::SceneDesc scene = Room(white, nullptr);
            for (bool nee : {true, false})
            {
                pt::ReferenceSettings settings = Settings(tasks, 32, bounces);
                settings.nextEventEstimation = nee;
                const double mean = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
                EXPECT(Near(mean, bounces + 1.0, 0.025, nee ? "room radiance (NEE+MIS)" : "room radiance (BSDF only)"));
            }
        }
    }
}

// NEE + MIS and pure BSDF sampling must converge to the same image.
TEST(Integrator_NeeAndBsdfSamplingAgree)
{
    TaskSystem tasks(4);
    const renderer::Material grey{.name = "Grey", .baseColor = {0.5f, 0.5f, 0.5f}, .roughness = 0.6f};
    const renderer::Material light{
        .name = "Light", .baseColor = {0.5f, 0.5f, 0.5f}, .roughness = 0.6f, .emissionNits = 10.0f};
    const renderer::SceneDesc scene = Room(grey, &light);
    pt::ReferenceSettings settings = Settings(tasks, 128, 6);
    const double withNee = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    settings.nextEventEstimation = false;
    const double bsdfOnly = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    EXPECT(withNee > 1.0);
    EXPECT(Near(withNee, bsdfOnly, 0.03, "NEE vs BSDF-only mean"));
}

// The furnace scene with white materials: every sphere must vanish into the 1000 nit background.
TEST(Integrator_WhiteFurnaceSpheresMatchEnvironment)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = renderer::MakeFurnaceScene();
    for (renderer::Mesh& mesh : scene.meshes)
    {
        mesh = renderer::MakeSphere(mesh.name, 0.2f, 24); // fewer triangles keep the test fast
    }
    pt::ReferenceSettings settings = Settings(tasks, 32, 16);
    settings.width = 48;
    settings.furnace = true;
    settings.applyExposure = true; // furnace exposure maps the environment to 0.18
    const Image image = pt::RenderReference(scene, scene.camera, settings);
    EXPECT(Near(MeanLuminance(image), 0.18, 0.01, "furnace mean"));
}

// Veach's MIS scene: lights from 3 cm to 90 cm reflected by smooth to rough plates. NEE with MIS
// and pure BSDF sampling are both unbiased, so they must agree on the plates, where each alone
// fails somewhere; this checks the MIS weights on glossy metal and spherical lights. Only the
// plate rows are averaged: the lights themselves are directly visible and identical in both.
// BSDF-only is heavy-tailed here (rarely hits the 3 cm light), so at 128 spp the two differ by about
// 2%; the tolerance catches weight errors, which bias by far more. At 32768 spp on the GPU they
// agree to 0.1%.
TEST(Integrator_VeachMisNeeAndBsdfSamplingAgree)
{
    TaskSystem tasks(4);
    const renderer::SceneDesc scene = *renderer::MakeBuiltinScene("veach_mis");
    pt::ReferenceSettings settings = Settings(tasks, 128, 4);
    settings.width = 48;
    settings.height = 27;
    auto plates = [](const Image& image)
    {
        double sum = 0.0;
        u32 count = 0;
        for (u32 y = image.height * 2 / 5; y < image.height * 9 / 10; ++y)
        {
            for (u32 x = 0; x < image.width; ++x)
            {
                const float* p = image.At(x, y);
                sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                ++count;
            }
        }
        return sum / count;
    };
    const double withNee = plates(pt::RenderReference(scene, scene.camera, settings));
    settings.nextEventEstimation = false;
    const double bsdfOnly = plates(pt::RenderReference(scene, scene.camera, settings));
    EXPECT(withNee > 0.0);
    EXPECT(Near(withNee, bsdfOnly, 0.08, "veach_mis plates, NEE vs BSDF-only"));
}

// The sun's disc is a light of its own: next event estimation of it, weighted by MIS against BSDF
// samples that reach it, must give the same image as BSDF sampling alone. The sun is made 12
// degrees wide here so that BSDF sampling finds it often enough to converge.
TEST(Integrator_SunNeeAndBsdfSamplingAgree)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = *renderer::MakeBuiltinScene("sunset");
    scene.sky.sunElevation = 40.0f;
    scene.sky.sunAngularRadius = 12.0f;
    pt::ReferenceSettings settings = Settings(tasks, 256, 3);
    settings.width = 32;
    settings.height = 18;
    const double withNee = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    settings.nextEventEstimation = false;
    const double bsdfOnly = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    EXPECT(withNee > 0.0);
    EXPECT(Near(withNee, bsdfOnly, 0.05, "sunset, NEE vs BSDF-only"));
}
