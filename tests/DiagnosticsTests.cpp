#include "TestHelpers.h"

#include "air/core/Profiler.h"
#include "air/renderer/Diagnostics.h"

using namespace air;
using namespace airtest;

TEST(Profiler_CollectsScopesFromTasksAndMainThread) {
    Profiler& profiler = Profiler::Get();
    profiler.SetEnabled(true);
    profiler.BeginFrame();
    {
        TaskSystem tasks(2);
        AIR_PROFILE_SCOPE("Test.Outer");
        { AIR_PROFILE_SCOPE("Test.Inner"); }
        tasks.Launch("Test.Task", [] {}).Wait();
    }
    profiler.EndFrame();
    profiler.SetEnabled(false);

    const CpuFrame frame = profiler.GetLastFrame();
    auto find = [&](std::string_view name) {
        return std::find_if(frame.events.begin(), frame.events.end(),
                            [&](const CpuEvent& e) { return e.name == name; });
    };
    EXPECT(find("Test.Outer") != frame.events.end());
    EXPECT(find("Test.Inner") != frame.events.end() && find("Test.Inner")->depth == find("Test.Outer")->depth + 1);
    EXPECT(find("Test.Task") != frame.events.end());
    EXPECT(renderer::ToJson(frame).Dump(0).find("\"Test.Outer\"") != std::string::npos);
}

TEST(FrameRunner_ProfileReportsPassTimingsAndMemory) {
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    {
        renderer::FeatureRegistry registry;
        renderer::RegisterBuiltinFeatures(registry);
        renderer::Scene scene(*device, renderer::MakeCornellBox());
        renderer::FrameRunner runner(*device, tasks, {.width = 32, .height = 16, .profile = true});
        EXPECT(runner.AddFeature("pathtracer", registry.Create("pathtracer")));

        const renderer::FrameResult result = runner.RenderFrame(scene.Get(), scene.Desc().camera);
        EXPECT(result.validationErrors.empty());
        EXPECT(result.passCpu.size() == 2 && result.passCpu[0].name == "PathTracer.Trace");
        // The Null GPU "finishes" immediately: the scopes of this frame are already resolved.
        EXPECT(result.gpu.scopes.size() == 2 && result.gpu.scopes[1].name == "PathTracer.Resolve");
        EXPECT(!result.memory.pools.empty() && result.memory.liveTextures > 0);

        const std::vector<rhi::ResourceInfo> resources = device->GetResourceList();
        EXPECT(std::any_of(resources.begin(), resources.end(),
                           [](const rhi::ResourceInfo& r) { return r.name == "FrameOutput"; }));

        const std::string json = result.ToJson().Dump(0);
        EXPECT(json.find("\"passCpu\"") != std::string::npos && json.find("\"gpu\"") != std::string::npos);
        EXPECT(json.find("\"pools\"") != std::string::npos);
    }
    device->WaitIdle();
}

TEST(Diagnostics_FormatBytes) {
    EXPECT(renderer::FormatBytes(512) == "512 B");
    EXPECT(renderer::FormatBytes(1536) == "1.50 KiB");
    EXPECT(renderer::FormatBytes(3ull << 30) == "3.00 GiB");
}
