// Built-in scenes: every one is valid and renders on the Null backend; the test scenes contain what
// they exist to test; the light list holds emitters only and follows material edits.

#include "TestHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace air;
using namespace airtest;

namespace
{

u32 EmissiveInstances(const renderer::SceneDesc& desc)
{
    u32 count = 0;
    for (const renderer::Instance& instance : desc.instances)
    {
        count += desc.materials[instance.material].emissionNits > 0.0f ? 1u : 0u;
    }
    return count;
}

u64 Triangles(const renderer::SceneDesc& desc)
{
    u64 count = 0;
    for (const renderer::Instance& instance : desc.instances)
    {
        count += desc.meshes[instance.mesh].indices.size() / 3;
    }
    return count;
}

} // namespace

TEST(Scene_BuiltinScenesValidateAndRender)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        TaskSystem tasks(2);
        renderer::FeatureRegistry registry;
        renderer::RegisterBuiltinFeatures(registry);
        for (const std::string& name : renderer::BuiltinSceneNames())
        {
            std::optional<renderer::SceneDesc> desc = renderer::MakeBuiltinScene(name);
            EXPECT(desc && renderer::Validate(*desc).empty());
            if (!desc)
            {
                continue;
            }
            renderer::Scene scene(*device, std::move(*desc));
            renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 18});
            EXPECT(runner.AddFeature("pathtracer", registry.Create("pathtracer")));
            const renderer::FrameResult result = runner.RenderFrame(scene.Get(), scene.Desc().camera);
            EXPECT(result.validationErrors.empty());
            EXPECT(scene.Get().lightCount > 0 || name == "furnace"); // only the furnace has no emitters
        }
    }
    device->WaitIdle();
    EXPECT(device->GetMemoryStats().liveBuffers == 0);
}

TEST(Scene_TestScenesContainWhatTheyTest)
{
    EXPECT(EmissiveInstances(*renderer::MakeBuiltinScene("veach_mis")) == 5); // 4 lights + the fill
    const renderer::SceneDesc field = *renderer::MakeBuiltinScene("sphere_field");
    EXPECT(field.instances.size() > 2000 && field.materials.size() >= 32 && EmissiveInstances(field) > 10);
    EXPECT(EmissiveInstances(*renderer::MakeBuiltinScene("many_lights")) >= 360);
    EXPECT(Triangles(*renderer::MakeBuiltinScene("terrain")) > 2'000'000);
    EXPECT(EmissiveInstances(*renderer::MakeBuiltinScene("door")) == 1);
    // Generated with a fixed seed: building twice gives the same scene.
    const renderer::SceneDesc again = *renderer::MakeBuiltinScene("sphere_field");
    EXPECT(again.instances.size() == field.instances.size());
    EXPECT(std::memcmp(again.instances.back().transform, field.instances.back().transform,
                       sizeof(field.instances.back().transform)) == 0);
}

// Only emissive instances are in the light list, and FirstLightIndices() indexes it consistently;
// making a material emissive at runtime adds its triangles.
TEST(Scene_LightListHoldsOnlyEmitters)
{
    renderer::SceneDesc desc = *renderer::MakeBuiltinScene("cornell_box");
    const std::vector<u32> first = renderer::FirstLightIndices(desc);
    const std::vector<gpu::LightTriangle> lights = renderer::BuildLightTriangles(desc);
    EXPECT(lights.size() == 2); // the ceiling panel: one quad
    for (size_t i = 0; i < desc.instances.size(); ++i)
    {
        const bool emits = desc.materials[desc.instances[i].material].emissionNits > 0.0f;
        EXPECT(emits == (first[i] != gpu::kInvalidIndex));
    }

    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        renderer::Scene scene(*device, std::move(desc));
        EXPECT(scene.Get().lightCount == 2);
        renderer::Material gold = scene.Desc().materials.back(); // the sphere
        gold.emissionNits = 100.0f;
        scene.SetMaterial(u32(scene.Desc().materials.size() - 1), gold);
        scene.Update();
        EXPECT(scene.Get().lightCount > 2);
    }
    device->WaitIdle();
}

// The light list's alias table selects every light with exactly its probability: entry i is drawn
// with 1/N and kept with aliasThreshold, otherwise it hands the draw to its alias.
TEST(Scene_LightAliasTableReproducesProbabilities)
{
    const renderer::SceneDesc desc = *renderer::MakeBuiltinScene("many_lights");
    const std::vector<gpu::LightTriangle> lights = renderer::BuildLightTriangles(desc);
    EXPECT(lights.size() > 10000);
    const double n = double(lights.size());
    std::vector<double> selected(lights.size(), 0.0);
    double total = 0.0;
    for (const gpu::LightTriangle& light : lights)
    {
        EXPECT(light.aliasThreshold >= 0.0f && light.aliasThreshold <= 1.0f && light.alias < lights.size());
        total += light.probability;
    }
    for (size_t i = 0; i < lights.size(); ++i)
    {
        selected[i] += double(lights[i].aliasThreshold) / n;
        selected[lights[i].alias] += (1.0 - double(lights[i].aliasThreshold)) / n;
    }
    double worst = 0.0;
    for (size_t i = 0; i < lights.size(); ++i)
    {
        worst = std::max(worst, std::abs(selected[i] - double(lights[i].probability)) / (1.0 / n));
    }
    EXPECT(std::abs(total - 1.0) < 1e-4);
    EXPECT(worst < 1e-5); // relative to the mean probability 1/N
}
