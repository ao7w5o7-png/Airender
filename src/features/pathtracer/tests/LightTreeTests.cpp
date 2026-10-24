// The light tree (renderer::BuildLightTree) and the integrator's sampling of it (Integrator.hlsli,
// run on the CPU through LightTreeProbe): every light has a leaf, the probabilities a point gives
// the lights add up to one, and sampling chooses each light with exactly that probability.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/features/pathtracer/ReferenceRenderer.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace air;
using renderer::float3;

namespace
{

// Points of the many_lights hall, with and without a receiver normal.
struct Probe
{
    float3 p;
    float3 n;
};

std::vector<Probe> HallProbes(const renderer::SceneDesc& scene)
{
    const float3 camera = scene.camera.position;
    return {{camera, float3(0.0f)},
            {camera + float3(0.0f, -1.0f, -3.0f), float3(0.0f, 1.0f, 0.0f)},
            {camera + float3(2.0f, 0.5f, -6.0f), float3(0.0f, 0.0f, 1.0f)},
            {camera + float3(-3.0f, 1.5f, -9.0f), float3(1.0f, 0.0f, 0.0f)}};
}

} // namespace

TEST(LightTree_HasALeafPerLight)
{
    const renderer::SceneDesc scene = airtest::BuiltinScene("many_lights");
    const pt::LightTreeProbe tree(scene);
    EXPECT(tree.GetLightCount() > 100);
    EXPECT(tree.GetNodeCount() == 2 * tree.GetLightCount() - 1); // a binary tree with one light per leaf
    for (u32 light = 0; light < tree.GetLightCount(); ++light)
    {
        EXPECT(tree.GetDepth(light) <= 64u);
    }
}

TEST(LightTree_SamplingMatchesThePmf)
{
    const renderer::SceneDesc scene = airtest::BuiltinScene("many_lights");
    const pt::LightTreeProbe tree(scene);
    const u32 count = tree.GetLightCount();
    for (const Probe& probe : HallProbes(scene))
    {
        std::vector<double> pmf(count);
        double total = 0.0;
        for (u32 light = 0; light < count; ++light)
        {
            pmf[light] = tree.Pmf(light, probe.p, probe.n);
            total += pmf[light];
        }
        // The lights that can light the point share at most 1: a node whose box may face the point
        // while neither child does (its cones are conservative) keeps the rest, and sampling finds
        // nothing there, as often as the shortfall. In the hall that is the far side of the LED
        // spheres, about a tenth.
        EXPECT(total > 0.85 && total < 1.0001);

        // Stratified uniforms: the frequency of each light converges like 1/N.
        constexpr u32 kSamples = 200000;
        std::vector<double> frequency(count);
        u32 none = 0;
        bool pmfMatches = true;
        for (u32 i = 0; i < kSamples; ++i)
        {
            float samplePmf = 0.0f;
            const u32 light = tree.Sample(probe.p, probe.n, (float(i) + 0.5f) / float(kSamples), &samplePmf);
            if (light >= count)
            {
                ++none;
                continue;
            }
            frequency[light] += 1.0 / kSamples;
            pmfMatches &= std::abs(samplePmf - pmf[light]) <= 1e-4 * std::max(pmf[light], 1e-3);
        }
        EXPECT(pmfMatches); // the pmf sampling reports is the one the trail gives
        double worst = 0.0;
        for (u32 light = 0; light < count; ++light)
        {
            worst = std::max(worst, std::abs(frequency[light] - pmf[light]));
        }
        if (worst > 2e-4)
        {
            std::fprintf(stderr, "  frequency differs from the pmf by %.5f\n", worst);
        }
        EXPECT(worst <= 2e-4);
        EXPECT(std::abs(double(none) / kSamples - (1.0 - total)) <= 1e-3);
    }
}

// Nearby lights get more samples than far ones of the same power: the point of the tree.
TEST(LightTree_PrefersNearbyLights)
{
    renderer::SceneDesc scene;
    scene.name = "two lamps";
    scene.materials.push_back({.baseColor = float3(0.0f), .emissionNits = 10.0f});
    for (const float z : {-1.0f, -20.0f})
    {
        scene.meshes.push_back(renderer::MakeQuad("Lamp", float3(-0.5f, 2.0f, z - 0.5f), float3(0.5f, 2.0f, z - 0.5f),
                                                  float3(0.5f, 2.0f, z + 0.5f), float3(-0.5f, 2.0f, z + 0.5f),
                                                  float3(0.0f, -1.0f, 0.0f)));
        scene.instances.push_back({.mesh = u32(scene.meshes.size() - 1), .material = 0});
    }
    const pt::LightTreeProbe tree(scene);
    const float3 p(0.0f, 0.0f, -1.0f); // under the first lamp
    const float3 n(0.0f, 1.0f, 0.0f);
    const float near = tree.Pmf(0, p, n) + tree.Pmf(1, p, n);
    const float far = tree.Pmf(2, p, n) + tree.Pmf(3, p, n);
    EXPECT(near > 0.9f && std::abs(near + far - 1.0f) < 1e-4f);
    // Facing down, the point cannot receive the lamps above it (the BSDFs only reflect); a point
    // above the lamps does not see their emitting side.
    EXPECT(tree.Pmf(0, p, float3(0.0f, -1.0f, 0.0f)) == 0.0f);
    float pmf = 1.0f;
    EXPECT(tree.Sample(float3(0.0f, 5.0f, -1.0f), float3(0.0f), 0.5f, &pmf) == ~0u && pmf == 0.0f);
}
