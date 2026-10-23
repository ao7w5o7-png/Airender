#include "TestHelpers.h"

#include "air/renderer/Parameters.h"

using namespace air;
using namespace airtest;

TEST(Parameters_ParseValidateAndVersion)
{
    renderer::ParameterSet params;
    u32 count = 1;
    float scale = 1.0f;
    bool enabled = false;
    params.Add("count", &count, "a count", 1, 10);
    params.Add("scale", &scale, "a scale", 0.0f, 2.0f);
    params.Add("enabled", &enabled, "a switch");

    std::string error;
    EXPECT(params.Set("count", "7") && count == 7);
    EXPECT(params.Set("scale", "1.5") && scale == 1.5f);
    EXPECT(params.Set("enabled", "true") && enabled);
    EXPECT(params.GetVersion() == 3);

    EXPECT(!params.Set("count", "11", &error) && count == 7);
    EXPECT(!params.Set("count", "x", &error));
    EXPECT(!params.Set("scale", "3", &error));
    EXPECT(!params.Set("enabled", "maybe", &error));
    EXPECT(!params.Set("unknown", "1", &error) && error.find("unknown") != std::string::npos);
    EXPECT(params.GetVersion() == 3);
    EXPECT(params.ToJson().Dump(0).find(
               "\"count\":{\"description\":\"a count\",\"max\":10,\"min\":1,\"type\":\"u32\",\"value\":7}") !=
           std::string::npos);
}

TEST(FeatureRegistry_BuiltinFeaturesAreRegistered)
{
    renderer::FeatureRegistry registry;
    renderer::RegisterBuiltinFeatures(registry);
    EXPECT(registry.Find("pathtracer") != nullptr);
    EXPECT(registry.Find("example") != nullptr);
    EXPECT(registry.Create("does-not-exist") == nullptr);
    for (const auto& info : registry.List())
    {
        EXPECT(!info.description.empty());
    }
}

TEST(FrameRunner_NoFeaturesStillValid)
{
    Headless h({});
    auto result = h.Frame();
    EXPECT(result.validationErrors.empty());
    EXPECT(result.graphStats.passCount == 0);
}

TEST(FrameRunner_MultipleFramesKeepStateConsistent)
{
    Headless h({"pathtracer", "example"});
    for (int i = 0; i < 5; ++i)
    {
        auto result = h.Frame();
        EXPECT(result.frameIndex == u64(i));
        EXPECT(result.validationErrors.empty());
    }
}

TEST(FrameRunner_ReleasesAllMemoryOnShutdown)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        TaskSystem tasks(2);
        renderer::FeatureRegistry registry;
        renderer::RegisterBuiltinFeatures(registry);
        renderer::Scene scene(*device, renderer::MakeCornellBox());
        renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 32});
        runner.AddFeature("pathtracer", registry.Create("pathtracer"));
        runner.AddFeature("example", registry.Create("example"));
        runner.RenderFrame(scene.Get(), scene.Desc().camera);
        runner.RenderFrame(scene.Get(), scene.Desc().camera);
    }
    device->WaitIdle();
    const auto stats = device->GetMemoryStats();
    EXPECT(stats.liveBuffers == 0 && stats.liveTextures == 0);
    for (const auto& location : stats.perLocation)
    {
        EXPECT(location.allocationCount == 0);
    }
}

// Rendering below the output size: the Scene stage gets its own smaller target, and without an
// Upscale feature FrameRunner fills the output with its bilinear upscale before the Post stage.
TEST(FrameRunner_RenderScaleUpscalesBeforePostFeatures)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        TaskSystem tasks(2);
        renderer::FeatureRegistry registry;
        renderer::RegisterBuiltinFeatures(registry);
        renderer::Scene scene(*device, renderer::MakeCornellBox());
        renderer::FrameRunner runner(*device, tasks,
                                     {.width = 64, .height = 36, .renderScale = 0.5f, .captureGraph = true});
        EXPECT(runner.GetRenderWidth() == 32 && runner.GetRenderHeight() == 18);
        // Added out of order on purpose: stages decide the order, not the call order.
        EXPECT(runner.AddFeature("tonemap", registry.Create("tonemap")));
        EXPECT(runner.AddFeature("pathtracer", registry.Create("pathtracer")));
        const renderer::FrameResult result = runner.RenderFrame(scene.Get(), scene.Desc().camera);
        EXPECT(result.validationErrors.empty());
        const size_t trace = result.graphText.find("PathTracer.Trace");
        const size_t upscale = result.graphText.find("FrameRunner.Upscale");
        const size_t tonemap = result.graphText.find("Tonemap");
        EXPECT(trace != std::string::npos && upscale != std::string::npos && tonemap != std::string::npos);
        EXPECT(trace < upscale && upscale < tonemap);
        EXPECT(result.graphText.find("FrameRunner.SceneColor") != std::string::npos);
    }
    device->WaitIdle();
    EXPECT(device->GetMemoryStats().liveTextures == 0);
}

// At native resolution the Scene stage renders straight into the output: no extra pass or texture.
TEST(FrameRunner_NativeResolutionHasNoUpscale)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    renderer::FeatureRegistry registry;
    renderer::RegisterBuiltinFeatures(registry);
    renderer::Scene scene(*device, renderer::MakeCornellBox());
    renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 32, .captureGraph = true});
    EXPECT(runner.AddFeature("pathtracer", registry.Create("pathtracer")));
    const renderer::FrameResult result = runner.RenderFrame(scene.Get(), scene.Desc().camera);
    EXPECT(result.validationErrors.empty());
    EXPECT(result.graphText.find("FrameRunner.Upscale") == std::string::npos);
    EXPECT(result.graphText.find("FrameRunner.SceneColor") == std::string::npos);
}

namespace
{

// Reads the diffuse albedo guide, like a denoiser would.
class GuideConsumer final : public renderer::RenderFeature
{
public:
    bool Initialize(const renderer::FeatureInitContext&) override { return true; }
    void AddPasses(renderer::FrameContext& context) override
    {
        found = context.blackboard.FindTexture(renderer::slots::kDiffuseAlbedo).has_value();
    }
    std::vector<std::string_view> ConsumedSlots() const override { return {renderer::slots::kDiffuseAlbedo}; }
    bool found = false;
};

} // namespace

// The path tracer writes its denoiser guides only when an active feature consumes them.
TEST(FrameRunner_GuidesOnlyForConsumers)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    renderer::FeatureRegistry registry;
    renderer::RegisterBuiltinFeatures(registry);
    renderer::Scene scene(*device, renderer::MakeCornellBox());
    {
        renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 18, .captureGraph = true});
        EXPECT(runner.AddFeature("pathtracer", registry.Create("pathtracer")));
        const renderer::FrameResult result = runner.RenderFrame(scene.Get(), scene.Desc().camera);
        EXPECT(result.graphText.find("PathTracer.DiffuseAlbedo") == std::string::npos);
    }
    {
        renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 18, .captureGraph = true});
        EXPECT(runner.AddFeature("pathtracer", registry.Create("pathtracer")));
        auto consumer = std::make_unique<GuideConsumer>();
        GuideConsumer* view = consumer.get();
        EXPECT(runner.AddFeature("consumer", std::move(consumer)));
        const renderer::FrameResult result = runner.RenderFrame(scene.Get(), scene.Desc().camera);
        EXPECT(view->found && result.graphText.find("PathTracer.DiffuseAlbedo") != std::string::npos);
        EXPECT(result.validationErrors.empty());
    }
    device->WaitIdle();
}
