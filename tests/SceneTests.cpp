// The light list holds emitters only, indexes them consistently and follows material edits.

#include "TestHelpers.h"

using namespace air;
using namespace airtest;

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
        if (emits)
        {
            EXPECT(lights[first[i]].instance == u32(i));
        }
    }
    EXPECT(lights.back().cdf == 1.0f);

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
