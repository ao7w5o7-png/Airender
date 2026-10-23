// aircli: headless driver for the renderer, built for humans and AI agents alike.
// Everything it does is available as text or JSON; exit codes are meaningful (0 ok, 1 failed check,
// 2 usage / setup error).
//
//   aircli render   [--backend null|d3d12|vulkan] [--features a,b] [--set feature.param=value]...
//                   [--frames N] [--width W] [--height H] [--render-scale S] [--shaders DIR]
//                   [--dump-graph out.dot] [--dump-commands] [--report out.json] [--profile] [--no-validation]
//                   [--no-async-compute] [--no-transient-pool] [--html profile.html]   (report of the last frame)
//                   [--scene NAME] [--output image.pfm|image.ppm]   (aircli scenes lists them)
//                   [--sun ELEVATION,AZIMUTH] [--aerosols DEPTH]   (physical sky, degrees)
//   aircli features                         list available features and their parameters (JSON)
//   aircli scenes                           list built-in scenes and their sizes (JSON)
//   aircli reference --output image.pfm [--scene NAME] [--width W] [--height H] [--spp N] [--bounces N]
//                    [--furnace] [--no-nee] [--light-tree-bounces N] [--sun E,A] [--aerosols D]
//   aircli compare REF TEST [--diff out.ppm] [--max-rmse X] [--threshold T] [--max-differing F]
//
// `render` fails (exit 1) if any frame produced a validation error.

#include "air/core/Image.h"
#include "air/core/Json.h"
#include "air/core/Log.h"
#include "air/core/Profiler.h"
#include "air/core/TaskSystem.h"
#include "air/features/pathtracer/ReferenceRenderer.h"
#include "air/renderer/Diagnostics.h"
#include "air/renderer/FrameRunner.h"
#include "air/renderer/ProfileReport.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"
#include "air/renderer/Transfer.h"
#include "air/rhi/RHIDevice.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace air;

namespace
{

constexpr int kExitOk = 0;
constexpr int kExitCheckFailed = 1;
constexpr int kExitUsage = 2;

int Usage()
{
    std::fputs("usage:\n"
               "  aircli render [--backend null|d3d12|vulkan] [--features a,b] [--set feature.param=value]...\n"
               "                [--frames N] [--width W] [--height H] [--render-scale S] [--shaders DIR]\n"
               "                [--dump-graph out.dot] [--dump-commands] [--report out.json] [--profile]\n"
               "                [--no-validation] [--no-async-compute] [--no-transient-pool] [--html profile.html]\n"
               "                [--scene NAME] [--output image.pfm|image.ppm]\n"
               "                [--sun ELEVATION,AZIMUTH] [--aerosols DEPTH]   (physical sky)\n"
               "  aircli reference --output image.pfm [--scene NAME] [--width W] [--height H] [--spp N]\n"
               "                [--bounces N] [--furnace] [--no-nee] [--light-tree-bounces N]\n"
               "                [--sun ELEVATION,AZIMUTH] [--aerosols DEPTH]\n"
               "  aircli features | scenes\n"
               "  aircli compare REF TEST [--diff out.ppm] [--max-rmse X] [--threshold T] [--max-differing F]\n",
               stderr);
    return kExitUsage;
}

std::vector<std::string> Split(std::string_view text, char separator)
{
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size())
    {
        const size_t end = std::min(text.find(separator, start), text.size());
        if (end > start)
        {
            parts.emplace_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    return parts;
}

// Small argument cursor: Next() returns the following argument or reports a usage error.
struct Args
{
    std::vector<std::string_view> items;
    size_t index = 0;
    bool error = false;

    bool Done() const { return index >= items.size(); }
    std::string_view Peek() const { return items[index]; }
    std::string_view Take() { return items[index++]; }
    std::string_view Next()
    {
        if (Done())
        {
            error = true;
            return {};
        }
        return Take();
    }
};

// ---------------------------------------------------------------------------
// features
// ---------------------------------------------------------------------------
int CommandFeatures()
{
    renderer::FeatureRegistry registry;
    renderer::RegisterBuiltinFeatures(registry);
    json::Value out = json::Value::MakeArray();
    for (const renderer::FeatureInfo& info : registry.List())
    {
        json::Value item = json::Value::MakeObject();
        item["name"] = info.name;
        item["description"] = info.description;
        item["parameters"] = info.create()->Parameters().ToJson();
        out.Push(std::move(item));
    }
    std::puts(out.Dump().c_str());
    return kExitOk;
}

// ---------------------------------------------------------------------------
// scenes
// ---------------------------------------------------------------------------
int CommandScenes()
{
    json::Value out = json::Value::MakeArray();
    for (const std::string& name : renderer::BuiltinSceneNames())
    {
        const renderer::SceneDesc desc = *renderer::MakeBuiltinScene(name);
        u64 triangles = 0;
        u32 emissive = 0;
        for (const renderer::Instance& instance : desc.instances)
        {
            triangles += desc.meshes[instance.mesh].indices.size() / 3;
            emissive += desc.materials[instance.material].emissionNits > 0.0f ? 1u : 0u;
        }
        json::Value item = json::Value::MakeObject();
        item["name"] = name;
        item["instances"] = u64(desc.instances.size());
        item["meshes"] = u64(desc.meshes.size());
        item["materials"] = u64(desc.materials.size());
        item["triangles"] = triangles;
        item["emissiveInstances"] = emissive;
        item["ev100"] = desc.camera.ev100;
        out.Push(std::move(item));
    }
    std::puts(out.Dump().c_str());
    return kExitOk;
}

// --sun ELEVATION,AZIMUTH (degrees) and --aerosols DEPTH: give any scene the physical sky (Sky.h).
struct SkyOptions
{
    std::optional<std::pair<float, float>> sun;
    std::optional<float> aerosols;

    // True when `arg` was one of these options.
    bool Parse(std::string_view arg, Args& args)
    {
        if (arg == "--sun")
        {
            const std::string value(args.Next());
            const size_t comma = value.find(',');
            sun = std::pair{std::stof(value.substr(0, comma)),
                            comma == std::string::npos ? 0.0f : std::stof(value.substr(comma + 1))};
            return true;
        }
        if (arg == "--aerosols")
        {
            aerosols = std::stof(std::string(args.Next()));
            return true;
        }
        return false;
    }

    void Apply(renderer::SceneDesc& scene) const
    {
        if (sun)
        {
            scene.sky.physical = true;
            scene.sky.sunElevation = sun->first;
            scene.sky.sunAzimuth = sun->second;
        }
        if (aerosols)
        {
            scene.sky.aerosolDepth = *aerosols;
        }
    }
};

// ---------------------------------------------------------------------------
// reference: the path tracer's estimator on the CPU, to validate GPU output with `compare`
// ---------------------------------------------------------------------------
int CommandReference(Args& args)
{
    std::string sceneName = "cornell_box";
    std::string outputPath;
    SkyOptions skyOptions;
    pt::ReferenceSettings settings;
    settings.width = 320;
    settings.height = 180;
    settings.samplesPerPixel = 256;
    while (!args.Done())
    {
        const std::string_view arg = args.Take();
        if (arg == "--scene")
        {
            sceneName = args.Next();
        }
        else if (arg == "--output")
        {
            outputPath = args.Next();
        }
        else if (arg == "--width")
        {
            settings.width = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--height")
        {
            settings.height = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--spp")
        {
            settings.samplesPerPixel = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--bounces")
        {
            settings.maxBounces = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--furnace")
        {
            settings.furnace = true;
        }
        else if (arg == "--no-nee")
        {
            settings.nextEventEstimation = false;
        }
        else if (arg == "--light-tree-bounces")
        {
            settings.lightTreeBounces = u32(std::stoul(std::string(args.Next()))); // as pathtracer.lightTreeBounces
        }
        else if (skyOptions.Parse(arg, args))
        {
        }
        else
        {
            AIR_LOG(Error, "aircli", "unknown option '{}'", arg);
            return Usage();
        }
    }
    std::optional<renderer::SceneDesc> scene = renderer::MakeBuiltinScene(sceneName);
    if (args.error || outputPath.empty() || !scene)
    {
        return Usage();
    }
    skyOptions.Apply(*scene);
    TaskSystem tasks;
    settings.tasks = &tasks;
    const auto start = std::chrono::steady_clock::now();
    const Image image = pt::RenderReference(*scene, scene->camera, settings);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const bool ppm = outputPath.ends_with(".ppm");
    if (!(ppm ? WritePpm(outputPath, image) : WritePfm(outputPath, image)))
    {
        AIR_LOG(Error, "aircli", "could not write '{}'", outputPath);
        return kExitCheckFailed;
    }
    json::Value report = json::Value::MakeObject();
    report["output"] = outputPath;
    report["scene"] = sceneName;
    report["seconds"] = seconds;
    std::puts(report.Dump().c_str());
    return kExitOk;
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------
int CommandCompare(Args& args)
{
    std::vector<std::string> files;
    std::string diffPath;
    ImageCompareOptions options;
    while (!args.Done())
    {
        const std::string_view arg = args.Take();
        if (arg == "--diff")
        {
            diffPath = args.Next();
        }
        else if (arg == "--max-rmse")
        {
            options.maxRmse = std::stof(std::string(args.Next()));
        }
        else if (arg == "--threshold")
        {
            options.pixelThreshold = std::stof(std::string(args.Next()));
        }
        else if (arg == "--max-differing")
        {
            options.maxDifferingFraction = std::stof(std::string(args.Next()));
        }
        else
        {
            files.emplace_back(arg);
        }
    }
    if (args.error || files.size() != 2)
    {
        return Usage();
    }
    const auto reference = ReadImage(files[0]);
    const auto test = ReadImage(files[1]);
    if (!reference || !test)
    {
        AIR_LOG(Error, "aircli", "could not read '{}' (supported: .pfm, .ppm)", !reference ? files[0] : files[1]);
        return kExitUsage;
    }
    const ImageCompareResult result = CompareImages(*reference, *test, options);
    if (!diffPath.empty() && !result.sizeMismatch)
    {
        WritePpm(diffPath, result.diff);
    }
    json::Value out = json::Value::MakeObject();
    out["passed"] = result.passed;
    out["sizeMismatch"] = result.sizeMismatch;
    out["rmse"] = result.rmse;
    out["psnr"] = result.psnr;
    out["maxAbsDiff"] = result.maxAbsDiff;
    out["differingPixels"] = result.differingPixels;
    out["differingFraction"] = result.differingFraction;
    out["summary"] = result.Summary();
    std::puts(out.Dump().c_str());
    return result.passed ? kExitOk : kExitCheckFailed;
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------
int CommandRender(Args& args)
{
    rhi::Backend backend = rhi::Backend::Null;
    bool validation = true;
    u32 frames = 4;
    renderer::FrameRunnerDesc desc;
    std::vector<std::string> featureNames = {"pathtracer"};
    std::vector<std::string> sets;
    std::string dumpGraph;
    std::string reportPath;
    std::string htmlPath;
    std::string sceneName = "cornell_box";
    std::string outputPath;
    SkyOptions skyOptions;
    bool dumpCommands = false;

    while (!args.Done())
    {
        const std::string_view arg = args.Take();
        if (arg == "--backend")
        {
            const std::string_view b = args.Next();
            backend = b == "d3d12" ? rhi::Backend::D3D12 : b == "vulkan" ? rhi::Backend::Vulkan : rhi::Backend::Null;
        }
        else if (arg == "--features")
        {
            featureNames = Split(args.Next(), ',');
        }
        else if (arg == "--set")
        {
            sets.emplace_back(args.Next());
        }
        else if (arg == "--frames")
        {
            frames = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--width")
        {
            desc.width = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--height")
        {
            desc.height = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--render-scale")
        {
            desc.renderScale = std::stof(std::string(args.Next())); // < 1: the scene renders smaller, dlss upscales
        }
        else if (arg == "--shaders")
        {
            desc.shaderRoot = std::string(args.Next());
        }
        else if (arg == "--dump-graph")
        {
            dumpGraph = args.Next();
        }
        else if (arg == "--dump-commands")
        {
            dumpCommands = true;
        }
        else if (arg == "--report")
        {
            reportPath = args.Next();
        }
        else if (arg == "--profile")
        {
            desc.profile = true;
        }
        else if (arg == "--no-validation")
        {
            validation = false; // no D3D12 debug layer / DRED: timings like a release run
        }
        else if (arg == "--no-async-compute")
        {
            desc.graphConfig.enableAsyncCompute = false; // async passes run on the graphics queue
        }
        else if (arg == "--html")
        {
            htmlPath = args.Next();
            desc.profile = true; // the report shows GPU timings
        }
        else if (arg == "--no-transient-pool")
        {
            desc.poolTransients = false; // a new transient heap and resources every frame
        }
        else if (arg == "--scene")
        {
            sceneName = args.Next();
        }
        else if (arg == "--output")
        {
            outputPath = args.Next();
        }
        else if (skyOptions.Parse(arg, args))
        {
        }
        else
        {
            AIR_LOG(Error, "aircli", "unknown option '{}'", arg);
            return Usage();
        }
    }
    if (args.error)
    {
        return Usage();
    }
    desc.captureCommands = dumpCommands;
    desc.captureGraph = true;
    Profiler::Get().SetEnabled(desc.profile);
    Profiler::Get().SetThreadName("Main");

    auto device = rhi::CreateDevice({.backend = backend, .enableValidation = validation});
    if (!device)
    {
        return kExitUsage;
    }
    TaskSystem tasks;
    AIR_LOG(Info, "aircli", "backend {} on '{}', {} worker threads", rhi::ToString(device->GetBackend()),
            device->GetCapabilities().adapterName, tasks.GetWorkerCount());

    json::Value report = json::Value::MakeObject();
    report["backend"] = rhi::ToString(device->GetBackend());
    report["adapter"] = device->GetCapabilities().adapterName;
    report["width"] = desc.width;
    report["height"] = desc.height;
    json::Value errors = json::Value::MakeArray();
    bool failed = false;

    {
        renderer::FeatureRegistry registry;
        renderer::RegisterBuiltinFeatures(registry);
        renderer::FrameRunner runner(*device, tasks, desc);
        std::optional<renderer::SceneDesc> sceneDesc = renderer::MakeBuiltinScene(sceneName);
        if (!sceneDesc)
        {
            AIR_LOG(Error, "aircli", "unknown scene '{}'", sceneName);
            return kExitUsage;
        }
        skyOptions.Apply(*sceneDesc);
        renderer::Scene scene(*device, std::move(*sceneDesc));
        const renderer::Camera camera = scene.Desc().camera;
        report["scene"] = sceneName;

        for (const std::string& name : featureNames)
        {
            auto feature = registry.Create(name);
            if (!feature)
            {
                AIR_LOG(Error, "aircli", "unknown feature '{}' (see `aircli features`)", name);
                return kExitUsage;
            }
            if (!runner.AddFeature(name, std::move(feature)))
            {
                return kExitUsage;
            }
        }
        for (const std::string& set : sets)
        {
            const size_t eq = set.find('=');
            std::string error;
            if (eq == std::string::npos)
            {
                error = "--set expects feature.param=value, got '" + set + "'";
            }
            else if (runner.SetParameter(std::string_view(set).substr(0, eq), std::string_view(set).substr(eq + 1),
                                         &error))
            {
                continue;
            }
            AIR_LOG(Error, "aircli", "{}", error);
            return kExitUsage;
        }

        json::Value frameReports = json::Value::MakeArray();
        json::Value commands = json::Value::MakeArray();
        std::shared_ptr<const renderer::FrameCapture> capture; // the last frame's, for --html
        for (u32 frame = 0; frame < frames; ++frame)
        {
            if (!htmlPath.empty() && frame + 1 == frames)
            {
                runner.RequestCapture();
            }
            Profiler::Get().BeginFrame();
            scene.Update();
            renderer::FrameResult result = runner.RenderFrame(scene.Get(), camera);
            Profiler::Get().EndFrame();
            if (result.capture)
            {
                capture = result.capture;
            }
            if (frame == 0)
            {
                std::fputs(result.graphText.c_str(), stdout);
                report["graph"] = result.graphText;
                if (!dumpGraph.empty())
                {
                    std::ofstream(dumpGraph) << result.graphviz;
                    AIR_LOG(Info, "aircli", "wrote {}", dumpGraph);
                }
            }
            for (const std::string& line : result.commands)
            {
                std::printf("frame %u: %s\n", frame, line.c_str());
                commands.Push(std::string("frame ") + std::to_string(frame) + ": " + line);
            }
            for (const std::string& e : result.validationErrors)
            {
                errors.Push(std::string("frame ") + std::to_string(frame) + ": " + e);
                failed = true;
            }
            frameReports.Push(result.ToJson());
        }
        report["frames"] = std::move(frameReports);
        if (!outputPath.empty())
        {
            // The frame output is left in SRVCompute by FrameRunner.
            const std::optional<Image> image =
                renderer::ReadbackTexture(*device, runner.GetOutput(), rhi::Access::SRVCompute);
            const bool ppm = outputPath.ends_with(".ppm");
            if (!image || !(ppm ? WritePpm(outputPath, *image) : WritePfm(outputPath, *image)))
            {
                AIR_LOG(Error, "aircli", "could not write '{}'", outputPath);
                errors.Push("could not write " + outputPath);
                failed = true;
            }
            else
            {
                AIR_LOG(Info, "aircli", "wrote {}", outputPath);
                report["output"] = outputPath;
            }
        }
        report["features"] = runner.DescribeFeatures();
        if (desc.profile)
        {
            device->WaitIdle(); // lets the last frames' GPU timings resolve
            report["gpuProfile"] = renderer::ToJson(device->GetGpuTimings());
            report["cpuProfile"] = renderer::ToJson(Profiler::Get().GetLastFrame());
            report["memory"] = renderer::ToJson(device->GetMemoryStats());
            report["processMemory"] = renderer::ToJson(GetProcessMemory());
        }
        if (capture)
        {
            // After WaitIdle the GPU timings are the last frame's: the captured one.
            const json::Value profile = renderer::MakeProfileReport({.capture = capture.get(),
                                                                     .gpu = device->GetGpuTimings(),
                                                                     .backend = rhi::ToString(device->GetBackend()),
                                                                     .adapter = device->GetCapabilities().adapterName,
                                                                     .scene = sceneName,
                                                                     .features = runner.DescribeFeatures()});
            if (renderer::WriteProfileHtml(htmlPath, profile))
            {
                AIR_LOG(Info, "aircli", "wrote {}", htmlPath);
                report["html"] = htmlPath;
            }
            else
            {
                AIR_LOG(Error, "aircli", "could not write '{}'", htmlPath);
                errors.Push("could not write " + htmlPath);
                failed = true;
            }
        }
        if (dumpCommands)
        {
            report["commands"] = std::move(commands);
        }
    }
    device->WaitIdle();

    json::Value memory = json::Value::MakeObject();
    const rhi::MemoryStats stats = device->GetMemoryStats();
    for (u32 i = 0; i < u32(rhi::MemoryLocation::Count); ++i)
    {
        const auto& s = stats.perLocation[i];
        json::Value& m = memory[rhi::ToString(rhi::MemoryLocation(i))];
        m["heaps"] = s.heapCount;
        m["reservedBytes"] = s.reservedBytes;
        m["usedBytes"] = s.usedBytes;
        m["allocations"] = s.allocationCount;
        if (s.allocationCount > 0)
        {
            errors.Push(std::format("leak: {} live allocations in {} after shutdown", s.allocationCount,
                                    rhi::ToString(rhi::MemoryLocation(i))));
            failed = true;
        }
    }
    report["memoryAfterShutdown"] = std::move(memory);
    report["errors"] = std::move(errors);
    report["passed"] = !failed;

    if (!reportPath.empty())
    {
        const std::filesystem::path path(reportPath);
        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream(path) << report.Dump() << '\n';
        AIR_LOG(Info, "aircli", "wrote {}", reportPath);
    }
    AIR_LOG(Info, "aircli", "{}", failed ? "FAILED (see errors in report)" : "OK");
    return failed ? kExitCheckFailed : kExitOk;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        args.items.emplace_back(argv[i]);
    }
    if (args.Done())
    {
        return Usage();
    }
    const std::string_view command = args.Take();
    if (command == "render")
    {
        return CommandRender(args);
    }
    if (command == "features")
    {
        return CommandFeatures();
    }
    if (command == "scenes")
    {
        return CommandScenes();
    }
    if (command == "reference")
    {
        return CommandReference(args);
    }
    if (command == "compare")
    {
        return CommandCompare(args);
    }
    return Usage();
}
