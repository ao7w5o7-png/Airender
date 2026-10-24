// Built-in scenes: every one is valid and renders on the Null backend; the test scenes contain what
// they exist to test; the light list holds emitters only and follows material edits; clouds stay
// above the geometry and their resources follow edits.

#include "TestHelpers.h"

#include <algorithm>
#include <cmath>

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
            // Every scene but the furnace has light: emitters, or the physical sky's sun.
            const renderer::float3 sun = scene.Get().sky.sunRadiance;
            EXPECT(scene.Get().lightCount > 0 || sun.x + sun.y + sun.z > 0.0f || name == "furnace");
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
    for (const char* name : {"sphere_field", "terrain", "sunset", "cumulus"})
    {
        const renderer::Sky sky = renderer::MakeBuiltinScene(name)->sky;
        EXPECT(sky.physical && sky.sunElevation > 0.0f); // daylight from the physical sky
    }
    EXPECT(renderer::MakeBuiltinScene("cumulus")->clouds.enabled);
    // Generated with a fixed seed: building twice gives the same scene.
    const renderer::SceneDesc again = *renderer::MakeBuiltinScene("sphere_field");
    EXPECT(again.instances.size() == field.instances.size());
    const auto& a = again.instances.back().transform;
    const auto& b = field.instances.back().transform;
    EXPECT(std::equal(&a[0][0], &a[0][0] + 12, &b[0][0]));
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

// Every emissive instance is one light group over its consecutive triangles: the Cornell box's panel
// (0.5 x 0.4 m at 3000 nits, facing down), and in the many-light hall one group per emitter whose
// triangles all name it.
TEST(Scene_LightGroupsGatherEachEmitter)
{
    const renderer::SceneDesc cornell = *renderer::MakeBuiltinScene("cornell_box");
    const std::vector<gpu::LightTriangle> lights = renderer::BuildLightTriangles(cornell);
    const std::vector<gpu::LightGroup> groups = renderer::BuildLightGroups(lights, cornell);
    EXPECT(groups.size() == 1);
    const gpu::LightGroup& panel = groups[0];
    EXPECT(panel.firstLight == 0 && panel.lightCount == 2);
    EXPECT(std::abs(panel.area - 0.2f) < 1e-5f);
    EXPECT(std::abs(panel.centre.x) < 1e-5f && std::abs(panel.centre.y - 1.995f) < 1e-5f &&
           std::abs(panel.centre.z) < 1e-5f);
    EXPECT(std::abs(panel.normalArea.y + 0.2f) < 1e-5f && std::abs(panel.normalArea.x) < 1e-6f);
    EXPECT(std::abs(panel.radius - std::sqrt(0.25f * 0.25f + 0.2f * 0.2f)) < 1e-5f);
    EXPECT(std::abs(panel.power - 3000.0f * 0.2f * 3.14159265f) < 0.1f);
    EXPECT(cornell.instances[panel.instance].name == "Light");

    const renderer::SceneDesc hall = *renderer::MakeBuiltinScene("many_lights");
    const std::vector<gpu::LightTriangle> hallLights = renderer::BuildLightTriangles(hall);
    const std::vector<gpu::LightGroup> hallGroups = renderer::BuildLightGroups(hallLights, hall);
    EXPECT(hallGroups.size() == EmissiveInstances(hall));
    double area = 0.0;
    for (u32 g = 0; g < hallGroups.size(); ++g)
    {
        const gpu::LightGroup& group = hallGroups[g];
        area += group.area;
        for (u32 i = group.firstLight; i < group.firstLight + group.lightCount; ++i)
        {
            EXPECT(hallLights[i].group == g);
        }
    }
    double lightArea = 0.0;
    for (const gpu::LightTriangle& light : hallLights)
    {
        lightArea += light.area;
    }
    EXPECT(std::abs(area / lightArea - 1.0) < 1e-6);
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

// Only camera rays look for clouds in front of a surface, so Validate keeps the cloud base above
// every instance.
TEST(Scene_CloudsStayAboveTheGeometry)
{
    renderer::SceneDesc desc = *renderer::MakeBuiltinScene("cumulus");
    EXPECT(renderer::Validate(desc).empty());
    desc.clouds.bottom = 5.0f; // below the roofs
    EXPECT(renderer::Validate(desc).find("above every instance") != std::string::npos);
    desc.clouds.bottom = 1500.0f;
    desc.clouds.top = 1550.0f; // too thin
    EXPECT(!renderer::Validate(desc).empty());
}

// Clouds are baked and uploaded when they change and released when they go.
TEST(Scene_CloudResourcesFollowEdits)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        renderer::Scene scene(*device, *renderer::MakeBuiltinScene("sunset"));
        EXPECT(scene.Get().clouds.enabled == 0 && !scene.Get().cloudBase.IsValid());
        renderer::Clouds clouds{.enabled = true};
        scene.SetClouds(clouds);
        scene.Update();
        EXPECT(scene.Get().clouds.enabled == 1 && scene.Get().cloudBase.IsValid() &&
               scene.Get().cloudDetail.IsValid() && scene.Get().cloudMajorants.IsValid());
        const u64 version = scene.Get().version;
        scene.SetClouds(clouds); // unchanged
        scene.Update();
        EXPECT(scene.Get().version == version);
        clouds.enabled = false;
        scene.SetClouds(clouds);
        scene.Update();
        EXPECT(scene.Get().clouds.enabled == 0 && !scene.Get().cloudBase.IsValid());
    }
    device->WaitIdle();
    EXPECT(device->GetMemoryStats().liveBuffers == 0 && device->GetMemoryStats().liveTextures == 0);
}

// Volumes are built and uploaded when they change and released when they go, each on its own: one
// that did not change keeps its resources. A file that cannot be read leaves none and says why; the
// GPU's table holds the volumes that loaded, with their bindless indices.
TEST(Scene_VolumeResourcesFollowEdits)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        renderer::Scene scene(*device, *renderer::MakeBuiltinScene("smoke"));
        scene.Update();
        const renderer::GpuScene& gpu = scene.Get();
        EXPECT(gpu.volumes.size() == 1 && gpu.volumes[0].enabled == 1 && gpu.volumeTable.IsValid() &&
               scene.VolumeError(0).empty());
        EXPECT(gpu.volumes[0].density != gpu.volumes[0].majorants); // bindless indices of two resources
        renderer::Volume volume = scene.Desc().volumes[0];
        const u64 version = gpu.version;
        const u32 density = gpu.volumes[0].density;
        scene.SetVolume(0, volume); // unchanged
        scene.Update();
        EXPECT(gpu.version == version);
        // A second plume beside the first: the first keeps its resources.
        renderer::Volume second = volume;
        second.position.x += 1.0f;
        second.seed = 7;
        scene.SetVolumes({volume, second});
        scene.Update();
        EXPECT(gpu.version != version && gpu.volumes.size() == 2 && gpu.volumes[0].density == density);
        EXPECT(std::abs(gpu.volumes[1].origin.x - gpu.volumes[0].origin.x - 1.0f) < 1e-5f);
        const u64 twoVolumes = gpu.version;
        second.file = "does/not/exist.vdb";
        scene.SetVolume(1, second);
        scene.Update();
        EXPECT(gpu.version != twoVolumes && gpu.volumes.size() == 1 && gpu.volumes[0].density == density);
        EXPECT(scene.VolumeError(0).empty() && scene.VolumeError(1).find("does/not/exist.vdb") != std::string::npos);
        volume.enabled = false;
        scene.SetVolumes({volume});
        scene.Update();
        EXPECT(gpu.volumes.empty() && !gpu.volumeTable.IsValid() && scene.VolumeError(0).empty() &&
               scene.VolumeError(1).empty());
    }
    device->WaitIdle();
    EXPECT(device->GetMemoryStats().liveBuffers == 0 && device->GetMemoryStats().liveTextures == 0);

    renderer::SceneDesc desc = *renderer::MakeBuiltinScene("smoke");
    EXPECT(renderer::Validate(desc).empty());
    desc.volumes[0].albedo = 1.5f;
    EXPECT(renderer::Validate(desc).find("volume 0") != std::string::npos);
    desc.volumes[0].albedo = 0.9f;
    desc.volumes[0].maxResolution = 4096; // beyond a 3D texture's 2048
    EXPECT(!renderer::Validate(desc).empty());
    desc.volumes[0].maxResolution = 256;
    desc.volumes.resize(gpu::kMaxVolumes + 1, desc.volumes[0]);
    EXPECT(renderer::Validate(desc).find("at most") != std::string::npos);
}
