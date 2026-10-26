#include "TestHelpers.h"

using namespace airtest;

// NRD runs on D3D12 only: elsewhere the feature refuses to run and says why, not crash.
TEST(Nrd_RefusesOnNullDevice)
{
    auto device = air::rhi::CreateDevice({.backend = air::rhi::Backend::Null});
    air::TaskSystem tasks(1);
    air::renderer::FeatureRegistry registry;
    air::renderer::RegisterBuiltinFeatures(registry);
    EXPECT(registry.Find("nrd") != nullptr);
    air::renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 32});
    EXPECT(!runner.AddFeature("nrd", registry.Create("nrd")));
    EXPECT(runner.FindFeature("nrd") == nullptr);
}
