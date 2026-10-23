#include "TestHelpers.h"

using namespace airtest;

// Without D3D12 and an RTX GPU the feature must refuse to run and say why, not crash.
TEST(Dlss_RefusesOnNullDevice)
{
    auto device = air::rhi::CreateDevice({.backend = air::rhi::Backend::Null});
    air::TaskSystem tasks(1);
    air::renderer::FeatureRegistry registry;
    air::renderer::RegisterBuiltinFeatures(registry);
    EXPECT(registry.Find("dlss") != nullptr);
    air::renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 32});
    EXPECT(!runner.AddFeature("dlss", registry.Create("dlss")));
    EXPECT(runner.FindFeature("dlss") == nullptr);
}
