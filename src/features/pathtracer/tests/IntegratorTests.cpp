// The path tracer's estimator (Integrator.hlsli) run on the CPU against analytic answers.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/features/pathtracer/ReferenceRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

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
    const renderer::SceneDesc scene = airtest::BuiltinScene("veach_mis");
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
    renderer::SceneDesc scene = airtest::BuiltinScene("sunset");
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

// Clouds scatter without absorbing (albedo 1 in the furnace), so in a uniform environment every
// path through them returns exactly the environment: they vanish, pixel by pixel.
TEST(Integrator_CloudsVanishInAWhiteFurnace)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("cumulus");
    scene.camera.pitch = 0.75f; // only sky and clouds in view
    pt::ReferenceSettings settings = Settings(tasks, 16, 4);
    settings.furnace = true;
    settings.applyExposure = true; // the environment maps to 0.18
    const Image image = pt::RenderReference(scene, scene.camera, settings);
    double lowest = 1e30;
    double highest = 0.0;
    for (u32 y = 0; y < image.height; ++y)
    {
        for (u32 x = 0; x < image.width; ++x)
        {
            const float* p = image.At(x, y);
            const double l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            lowest = std::min(lowest, l);
            highest = std::max(highest, l);
        }
    }
    EXPECT(Near(lowest, 0.18, 1e-3, "darkest pixel"));
    EXPECT(Near(highest, 0.18, 1e-3, "brightest pixel"));
}

// Inside the clouds the sun is reached two ways, by next event estimation (its transmittance by
// ratio tracking) and by phase sampling (delta tracking decides whether it gets through), weighed
// by MIS: both together must give what phase sampling alone gives. The sun is 12 degrees wide so
// that phase sampling finds it often enough.
TEST(Integrator_CloudSunNeeAndPhaseSamplingAgree)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("cumulus");
    scene.camera.pitch = 0.6f;
    scene.sky.sunElevation = 50.0f;
    scene.sky.sunAngularRadius = 12.0f;
    for (bool exact : {false, true})
    {
        scene.clouds.exactPhase = exact;
        pt::ReferenceSettings settings = Settings(tasks, 512, 2);
        settings.width = 16;
        settings.height = 16;
        const double withNee = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
        settings.nextEventEstimation = false;
        const double phaseOnly = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
        EXPECT(withNee > 0.0);
        EXPECT(Near(withNee, phaseOnly, 0.05,
                    exact ? "clouds (exact phase), NEE vs phase sampling" : "clouds, NEE vs phase sampling"));
    }
}

// A volume that scatters without absorbing (albedo 1 in the furnace) returns the environment along
// every path through it, pixel by pixel: no energy is lost or made at its scattering events. Its
// only surface is a speck far below, so that no path meets the BSDFs' Russian roulette.
TEST(Integrator_VolumeVanishesInAWhiteFurnace)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
    scene.instances.resize(1);
    scene.meshes[scene.instances[0].mesh] =
        renderer::MakeQuad("Speck", {-0.01f, -1000.0f, 0.01f}, {0.01f, -1000.0f, 0.01f}, {0.01f, -1000.0f, -0.01f},
                           {-0.01f, -1000.0f, -0.01f}, {0.0f, 1.0f, 0.0f}); // instance 0, the yard, is not moved
    scene.volumes[0].extinction = 8.0f;
    scene.camera.position = {0.0f, 1.9f, 3.0f};
    scene.camera.yaw = 0.0f;
    scene.camera.pitch = 0.0f;
    scene.camera.verticalFov = 1.0f;
    pt::ReferenceSettings settings = Settings(tasks, 16, 4);
    settings.furnace = true;
    settings.volumeEvents = 4096;  // walks longer than this would lose their light
    settings.applyExposure = true; // the environment maps to 0.18
    const Image image = pt::RenderReference(scene, scene.camera, settings);
    double lowest = 1e30;
    double highest = 0.0;
    for (u32 y = 0; y < image.height; ++y)
    {
        for (u32 x = 0; x < image.width; ++x)
        {
            const float* p = image.At(x, y);
            const double l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            lowest = std::min(lowest, l);
            highest = std::max(highest, l);
        }
    }
    EXPECT(Near(lowest, 0.18, 1e-3, "darkest pixel"));
    EXPECT(Near(highest, 0.18, 1e-3, "brightest pixel"));
}

// Overlapping volumes (each tracked on its own, the nearest collision scattering with its own phase
// function) neither lose nor make energy either: two plumes that cross, of different densities and
// anisotropies, vanish in the white furnace.
TEST(Integrator_OverlappingVolumesVanishInAWhiteFurnace)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
    scene.instances.resize(1);
    scene.meshes[scene.instances[0].mesh] =
        renderer::MakeQuad("Speck", {-0.01f, -1000.0f, 0.01f}, {0.01f, -1000.0f, 0.01f}, {0.01f, -1000.0f, -0.01f},
                           {-0.01f, -1000.0f, -0.01f}, {0.0f, 1.0f, 0.0f});
    scene.volumes[0].extinction = 6.0f;
    renderer::Volume second = scene.volumes[0];
    second.position.x += 0.4f;
    second.position.y += 0.3f;
    second.extinction = 10.0f;
    second.anisotropy = -0.3f;
    second.seed = 11;
    scene.volumes.push_back(second);
    scene.camera.position = {0.2f, 1.9f, 3.0f};
    scene.camera.yaw = 0.0f;
    scene.camera.pitch = 0.0f;
    scene.camera.verticalFov = 1.0f;
    pt::ReferenceSettings settings = Settings(tasks, 16, 4);
    settings.furnace = true;
    settings.volumeEvents = 4096;
    settings.applyExposure = true;
    const Image image = pt::RenderReference(scene, scene.camera, settings);
    double lowest = 1e30;
    double highest = 0.0;
    for (u32 y = 0; y < image.height; ++y)
    {
        for (u32 x = 0; x < image.width; ++x)
        {
            const float* p = image.At(x, y);
            const double l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            lowest = std::min(lowest, l);
            highest = std::max(highest, l);
        }
    }
    EXPECT(Near(lowest, 0.18, 1e-3, "darkest pixel"));
    EXPECT(Near(highest, 0.18, 1e-3, "brightest pixel"));
}

// Two volumes in the same place add up: two copies of the plume at half the extinction give the
// image of one at the full extinction (the nearest of two collisions in half as dense media is
// distributed as the first collision in the whole, and transmittances multiply). Lit from the side
// by the sun, so that both its own shadow and multiple scattering show.
TEST(Integrator_TwoHalvesOfAVolumeLookLikeTheWhole)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
    for (renderer::Material& material : scene.materials)
    {
        material.emissionNits = 0.0f;
    }
    scene.sky.sunElevation = 25.0f;
    scene.sky.sunAzimuth = -90.0f;
    scene.camera.position = {0.0f, 1.6f, 2.2f};
    scene.camera.yaw = 0.0f;
    scene.camera.pitch = 0.03f;
    scene.camera.verticalFov = 0.9f;
    pt::ReferenceSettings settings = Settings(tasks, 512, 3);
    settings.width = 16;
    settings.height = 16;
    const double whole = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    scene.volumes[0].extinction *= 0.5f;
    scene.volumes.push_back(scene.volumes[0]);
    const double halves = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
    EXPECT(whole > 0.0);
    EXPECT(Near(halves, whole, 0.02, "two halves vs the whole"));
}

// In and behind a volume, lights are reached two ways: by next event estimation (their transmittance
// by ratio tracking, from the medium's scattering events and from the surfaces) and by phase or BSDF
// sampling (delta tracking decides whether the ray gets through), weighed by MIS: both together must
// give what sampling alone gives. Three views, lit by the sun alone (made 12 degrees wide so that
// sampling finds it often enough) or by a 3 m lamp (a triangle light): the plume's shadow on the
// wall, the plume lit from the side (half of it in its own shadow), and the plume and its shadow on
// the ground under the lamp. Sampling alone is the noisier of the two, about 1% here; at 16 times
// the samples they agree to 0.3%. Removing the transmittance from any of the four NEE estimates fails
// the test.
TEST(Integrator_VolumeNeeAndPhaseSamplingAgree)
{
    TaskSystem tasks(4);
    constexpr const char* kViews[] = {"its shadow in the sun", "the plume in the sun", "the plume under a lamp"};
    for (int view = 0; view < 3; ++view)
    {
        renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
        for (renderer::Material& material : scene.materials)
        {
            material.emissionNits = 0.0f;
        }
        u32 spp = 1024;
        if (view == 0)
        {
            scene.sky.sunElevation = 25.0f;
            scene.sky.sunAzimuth = 35.0f;
            scene.sky.sunAngularRadius = 12.0f;
            scene.camera.position = {-1.2f, 1.3f, 1.8f};
            scene.camera.yaw = 0.25f;
            scene.camera.pitch = -0.05f;
            spp = 512;
        }
        else if (view == 1)
        {
            scene.sky.sunElevation = 25.0f;
            scene.sky.sunAzimuth = -90.0f; // from the left
            scene.sky.sunAngularRadius = 12.0f;
            scene.camera.position = {0.0f, 1.6f, 2.2f};
            scene.camera.yaw = 0.0f;
            scene.camera.pitch = 0.03f;
        }
        else
        {
            scene.sky.physical = false; // a black environment
            scene.materials.push_back({.name = "Lamp", .baseColor = {0.0f, 0.0f, 0.0f}, .emissionNits = 20000.0f});
            scene.meshes.push_back(renderer::MakeQuad("Lamp", {-0.5f, 4.5f, -1.5f}, {2.5f, 4.5f, -1.5f},
                                                      {2.5f, 4.5f, 1.5f}, {-0.5f, 4.5f, 1.5f}, {0.0f, -1.0f, 0.0f}));
            renderer::Instance lamp;
            lamp.name = "Lamp";
            lamp.mesh = u32(scene.meshes.size() - 1);
            lamp.material = u32(scene.materials.size() - 1);
            scene.instances.push_back(lamp);
            scene.camera.position = {-1.0f, 2.4f, 2.4f};
            scene.camera.yaw = 0.0f;
            scene.camera.pitch = -0.6f;
        }
        scene.camera.verticalFov = 0.9f;
        pt::ReferenceSettings settings = Settings(tasks, spp, 3);
        settings.width = 16;
        settings.height = 16;
        const double withNee = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
        settings.nextEventEstimation = false;
        const double samplingOnly = MeanLuminance(pt::RenderReference(scene, scene.camera, settings));
        EXPECT(withNee > 0.0);
        EXPECT(Near(withNee, samplingOnly, 0.04, (std::string(kViews[view]) + ", NEE vs sampling").c_str()));
    }
}
