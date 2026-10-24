#pragma once

// Shared helpers for renderer and feature tests: a headless (Null backend) environment that runs
// features through the same FrameRunner the app uses, plus golden-file comparison.

#include "TestFramework.h"

#include "air/core/Assert.h"
#include "air/core/TaskSystem.h"
#include "air/renderer/FrameRunner.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace airtest
{

inline bool ContainsLine(const std::vector<std::string>& lines, std::string_view needle)
{
    return std::any_of(lines.begin(), lines.end(),
                       [&](const std::string& l) { return l.find(needle) != std::string::npos; });
}

// A built-in scene the test names: it must exist.
inline air::renderer::SceneDesc BuiltinScene(std::string_view name)
{
    std::optional<air::renderer::SceneDesc> scene = air::renderer::MakeBuiltinScene(name);
    AIR_CHECK(scene.has_value(), "no built-in scene '{}'", name);
    return std::move(*scene);
}

inline size_t CountLines(const std::vector<std::string>& lines, std::string_view needle)
{
    return size_t(std::count_if(lines.begin(), lines.end(),
                                [&](const std::string& l) { return l.find(needle) != std::string::npos; }));
}

// Null device + task system + the Cornell box scene + FrameRunner with the named features.
class Headless
{
public:
    explicit Headless(const std::vector<std::string>& features, air::u32 width = 64, air::u32 height = 32,
                      bool captureGraph = true)
        : device(air::rhi::CreateDevice({.backend = air::rhi::Backend::Null})), tasks(2)
    {
        air::renderer::FeatureRegistry registry;
        air::renderer::RegisterBuiltinFeatures(registry);
        scene = std::make_unique<air::renderer::Scene>(*device, air::renderer::MakeCornellBox());
        camera = scene->Desc().camera;
        runner = std::make_unique<air::renderer::FrameRunner>(
            *device, tasks,
            air::renderer::FrameRunnerDesc{
                .width = width, .height = height, .captureCommands = true, .captureGraph = captureGraph});
        for (const std::string& name : features)
        {
            auto feature = registry.Create(name);
            EXPECT(feature != nullptr);
            if (feature)
            {
                EXPECT(runner->AddFeature(name, std::move(feature)));
            }
        }
    }

    ~Headless()
    {
        runner.reset();
        scene.reset();
        device->WaitIdle();
    }

    air::renderer::FrameResult Frame()
    {
        scene->Update();
        return runner->RenderFrame(scene->Get(), camera);
    }

    std::unique_ptr<air::rhi::Device> device;
    air::TaskSystem tasks;
    std::unique_ptr<air::renderer::Scene> scene;
    air::renderer::Camera camera;
    std::unique_ptr<air::renderer::FrameRunner> runner;
};

// Compares `actual` against tests/golden/<name>. Set AIR_UPDATE_GOLDEN=1 to (re)write the files,
// then review the diff with git like any other code change.
inline bool CheckGolden(const std::string& name, const std::string& actual)
{
    const std::filesystem::path path = std::filesystem::path(AIR_SOURCE_DIR) / "tests" / "golden" / name;
    const char* update = std::getenv("AIR_UPDATE_GOLDEN"); // NOLINT(concurrency-mt-unsafe): test thread only
    if (update && std::string(update) == "1")
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << actual;
        std::printf("  updated golden %s\n", path.string().c_str());
        return true;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::fprintf(stderr, "  missing golden %s (run with AIR_UPDATE_GOLDEN=1 to create it)\n",
                     path.string().c_str());
        return false;
    }
    std::stringstream expected;
    expected << file.rdbuf();
    if (expected.str() == actual)
    {
        return true;
    }
    // Report the first differing line so the failure is actionable without a diff tool.
    std::istringstream a(expected.str()), b(actual);
    std::string la, lb;
    int line = 1;
    while (true)
    {
        const bool ha = bool(std::getline(a, la));
        const bool hb = bool(std::getline(b, lb));
        if (!ha && !hb)
        {
            break;
        }
        if (!ha || !hb || la != lb)
        {
            std::fprintf(stderr, "  golden %s differs at line %d\n    expected: %s\n    actual:   %s\n", name.c_str(),
                         line, ha ? la.c_str() : "<end of file>", hb ? lb.c_str() : "<end of file>");
            break;
        }
        ++line;
    }
    std::fprintf(stderr, "  if the change is intended: AIR_UPDATE_GOLDEN=1 ./air_tests Golden\n");
    return false;
}

} // namespace airtest
