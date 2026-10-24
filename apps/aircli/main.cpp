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
//                   [--clouds COVERAGE[,BOTTOM,TOP]] [--clouds-exact]   (cloud layer; 0: none)
//                   [--camera EX,EY,EZ,TX,TY,TZ[,FOV]] [--ev EV100]   (eye, target, degrees)
//                   [--volume FILE.vdb|plume|none[,EXTINCTION[,ALBEDO[,ANISOTROPY]]]] [--volume-grid NAME]
//                   [--volume-at X,Y,Z[,SIZE]]...   (smoke or clouds of voxels among the surfaces)
//   aircli features                         list available features and their parameters (JSON)
//   aircli scenes                           list built-in scenes and their sizes (JSON)
//   aircli reference --output image.pfm [--scene NAME] [--width W] [--height H] [--spp N] [--bounces N]
//                    [--furnace] [--no-nee] [--light-tree-bounces N] [--sun E,A] [--aerosols D]
//                    [--clouds C[,B,T]] [--clouds-exact] [--cloud-events N] [--camera ...] [--ev EV100]
//                    [--volume ...] [--volume-grid NAME] [--volume-at ...] [--volume-events N]
//                    [--volume-march [--frames N] [--probe-frames N] [--volume-step S]
//                    [--volume-shadow-size N] [--volume-lights N]]   (the real-time volumes' method)
//   aircli vdb FILE.vdb [--read GRID] [--max-resolution N]   the grids of an OpenVDB file (JSON); --read
//                    reads one as a volume would and reports its voxels
//   aircli compare REF TEST [--diff out.ppm] [--max-rmse X] [--threshold T] [--max-differing F]
//   aircli mlp-check [--backend d3d12] [--shaders DIR] [--samples N]   the GPU's neural network kernels
//                    (float32 and linear algebra) against the CPU's (JSON; exit 1 when they disagree)
//
// `render` fails (exit 1) if any frame produced a validation error.

#include "air/core/Assert.h"
#include "air/core/Image.h"
#include "air/core/Json.h"
#include "air/core/Log.h"
#include "air/core/Profiler.h"
#include "air/core/TaskSystem.h"
#include "air/features/pathtracer/ReferenceRenderer.h"
#include "air/renderer/Diagnostics.h"
#include "air/renderer/FrameRunner.h"
#include "air/renderer/NeuralNetwork.h"
#include "air/renderer/ProfileReport.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"
#include "air/renderer/Transfer.h"
#include "air/renderer/Vdb.h"
#include "air/rhi/RHIDevice.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
               "                [--clouds COVERAGE[,BOTTOM,TOP]] [--clouds-exact]   (cloud layer; 0: none)\n"
               "                [--camera EX,EY,EZ,TX,TY,TZ[,FOV]] [--ev EV100]   (eye, target, degrees)\n"
               "                [--volume FILE.vdb|plume|none[,EXTINCTION[,ALBEDO[,ANISOTROPY]]]]\n"
               "                [--volume-grid NAME] [--volume-at X,Y,Z[,SIZE]]...   (smoke or clouds of voxels)\n"
               "  aircli reference --output image.pfm [--scene NAME] [--width W] [--height H] [--spp N]\n"
               "                [--bounces N] [--furnace] [--no-nee] [--light-tree-bounces N]\n"
               "                [--sun ELEVATION,AZIMUTH] [--aerosols DEPTH]\n"
               "                [--clouds COVERAGE[,BOTTOM,TOP]] [--clouds-exact] [--cloud-events N]\n"
               "                [--camera EX,EY,EZ,TX,TY,TZ[,FOV]] [--ev EV100]\n"
               "                [--volume ...] [--volume-grid NAME] [--volume-at ...] [--volume-events N]\n"
               "                [--volume-march [--frames N] [--probe-frames N] [--volume-step S]\n"
               "                [--volume-shadow-size N] [--volume-lights N]]\n"
               "  aircli features | scenes\n"
               "  aircli vdb FILE.vdb [--read GRID] [--max-resolution N]\n"
               "  aircli compare REF TEST [--diff out.ppm] [--max-rmse X] [--threshold T] [--max-differing F]\n"
               "  aircli mlp-check [--backend d3d12] [--shaders DIR] [--samples N]\n",
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
        const std::optional<renderer::SceneDesc> found = renderer::MakeBuiltinScene(name);
        AIR_CHECK(found.has_value(), "scene '{}' is listed but cannot be made", name);
        const renderer::SceneDesc& desc = *found;
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

// Changes to a built-in scene:
//   --sun ELEVATION,AZIMUTH (degrees) and --aerosols DEPTH give it the physical sky (Sky.h);
//   --clouds COVERAGE[,BOTTOM,TOP] (metres) adds a cloud layer (Clouds.h), 0 removes it; --clouds-exact
//   keeps their phase function's diffraction peak;
//   --camera EX,EY,EZ,TX,TY,TZ[,FOV] puts the camera at the eye looking at the target (vertical field
//   of view in degrees), --ev EV100 sets its exposure;
//   --volume FILE.vdb|plume|none[,EXTINCTION[,ALBEDO[,ANISOTROPY]]] puts a volume (Volume.h) in it: a
//   grid of an OpenVDB file (--volume-grid NAME, else "density" or its first float grid) or the
//   procedural plume of smoke; none removes them all. The n-th --volume sets the scene's n-th volume
//   (its other settings stay) or adds one. --volume-at X,Y,Z[,SIZE] places the last one given (before
//   any --volume: the first): with SIZE its longest side is scaled to SIZE metres and its bottom
//   centred on the point, else the point moves the file's own placement; --volume-grid likewise.
struct SceneOptions
{
    std::optional<std::pair<float, float>> sun;
    std::optional<float> aerosols;
    std::vector<std::string> clouds;
    bool cloudsExact = false;
    std::vector<std::string> camera;
    std::optional<float> ev100;
    struct VolumeOption
    {
        std::vector<std::string> spec; // empty: options given before any --volume
        std::optional<std::string> grid;
        std::vector<std::string> at;
    };
    std::vector<VolumeOption> volumes;

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
        if (arg == "--clouds")
        {
            clouds = Split(args.Next(), ',');
            return true;
        }
        if (arg == "--clouds-exact")
        {
            cloudsExact = true;
            return true;
        }
        if (arg == "--camera")
        {
            camera = Split(args.Next(), ',');
            return true;
        }
        if (arg == "--ev")
        {
            ev100 = std::stof(std::string(args.Next()));
            return true;
        }
        if (arg == "--volume")
        {
            if (volumes.empty() || !volumes.back().spec.empty())
            {
                volumes.emplace_back();
            }
            volumes.back().spec = Split(args.Next(), ',');
            return true;
        }
        if (arg == "--volume-grid" || arg == "--volume-at")
        {
            if (volumes.empty())
            {
                volumes.emplace_back();
            }
            if (arg == "--volume-grid")
            {
                volumes.back().grid = std::string(args.Next());
            }
            else
            {
                volumes.back().at = Split(args.Next(), ',');
            }
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
        if (!clouds.empty())
        {
            scene.clouds.coverage = std::stof(clouds[0]);
            scene.clouds.enabled = scene.clouds.coverage > 0.0f;
            if (clouds.size() >= 3)
            {
                scene.clouds.bottom = std::stof(clouds[1]);
                scene.clouds.top = std::stof(clouds[2]);
            }
            // Clouds in a uniform environment would have no sun to light them.
            scene.sky.physical = scene.sky.physical || scene.clouds.enabled;
        }
        scene.clouds.exactPhase = scene.clouds.exactPhase || cloudsExact;
        if (camera.size() >= 6)
        {
            const hlsl::float3 eye(std::stof(camera[0]), std::stof(camera[1]), std::stof(camera[2]));
            const hlsl::float3 d =
                hlsl::normalize(hlsl::float3(std::stof(camera[3]), std::stof(camera[4]), std::stof(camera[5])) - eye);
            scene.camera.position = eye;
            scene.camera.yaw = std::atan2(-d.x, -d.z); // yaw 0, pitch 0 looks down -z
            scene.camera.pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
            if (camera.size() >= 7)
            {
                scene.camera.verticalFov = std::stof(camera[6]) * 3.14159265f / 180.0f;
            }
        }
        if (ev100)
        {
            scene.camera.ev100 = *ev100;
        }
        size_t next = 0; // the scene volume the next --volume sets
        for (const VolumeOption& option : volumes)
        {
            if (!option.spec.empty() && option.spec[0] == "none")
            {
                scene.volumes.clear();
                next = 0;
                continue;
            }
            if (next >= scene.volumes.size())
            {
                if (option.spec.empty())
                {
                    continue; // placement for a volume the scene does not have
                }
                scene.volumes.emplace_back();
            }
            renderer::Volume& volume = scene.volumes[next++];
            const std::vector<std::string>& spec = option.spec;
            if (!spec.empty())
            {
                volume.enabled = true;
                volume.file = spec[0] == "plume" ? std::string() : spec[0];
                if (spec.size() >= 2)
                {
                    volume.extinction = std::stof(spec[1]);
                }
                if (spec.size() >= 3)
                {
                    volume.albedo = std::stof(spec[2]);
                }
                if (spec.size() >= 4)
                {
                    volume.anisotropy = std::stof(spec[3]);
                }
            }
            if (option.grid)
            {
                volume.grid = *option.grid;
            }
            if (option.at.size() >= 3)
            {
                volume.position = {std::stof(option.at[0]), std::stof(option.at[1]), std::stof(option.at[2])};
                volume.size = option.at.size() >= 4 ? std::stof(option.at[3]) : 0.0f;
            }
        }
        scene.volumes.resize(std::min<size_t>(scene.volumes.size(), gpu::kMaxVolumes));
    }
};

// ---------------------------------------------------------------------------
// reference: the path tracer's estimator on the CPU, to validate GPU output with `compare`; with
// --volume-march the real-time volumes' (pt::RenderVolumeMarchReference), `--spp` paths per pixel and
// frame
// ---------------------------------------------------------------------------
int CommandReference(Args& args)
{
    std::string sceneName = "cornell_box";
    std::string outputPath;
    SceneOptions sceneOptions;
    pt::ReferenceSettings settings;
    settings.width = 320;
    settings.height = 180;
    settings.samplesPerPixel = 0; // 256, or 16 a frame with --volume-march
    bool volumeMarch = false;
    pt::VolumeMarchReferenceSettings march;
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
        else if (arg == "--cloud-events")
        {
            settings.cloudEvents = u32(std::stoul(std::string(args.Next()))); // as pathtracer.cloudEvents
        }
        else if (arg == "--volume-events")
        {
            settings.volumeEvents = u32(std::stoul(std::string(args.Next()))); // as pathtracer.volumeEvents
        }
        else if (arg == "--light-tree-bounces")
        {
            settings.lightTreeBounces = u32(std::stoul(std::string(args.Next()))); // as pathtracer.lightTreeBounces
        }
        else if (arg == "--volume-march")
        {
            volumeMarch = true;
        }
        else if (arg == "--frames")
        {
            march.frames = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--probe-frames")
        {
            march.probeFrames = u32(std::stoul(std::string(args.Next())));
        }
        else if (arg == "--volume-step")
        {
            march.march.step = std::stof(std::string(args.Next())); // as pathtracer.volumeStep
        }
        else if (arg == "--volume-shadow-size")
        {
            march.march.shadowSize = u32(std::stoul(std::string(args.Next()))); // as pathtracer.volumeShadowSize
        }
        else if (arg == "--volume-lights")
        {
            march.march.lights = u32(std::stoul(std::string(args.Next()))); // as pathtracer.volumeLights
        }
        else if (sceneOptions.Parse(arg, args))
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
    sceneOptions.Apply(*scene);
    if (volumeMarch && scene->volumes.empty())
    {
        AIR_LOG(Error, "aircli", "--volume-march: scene '{}' has no volumes", sceneName);
        return kExitUsage;
    }
    if (settings.samplesPerPixel == 0)
    {
        settings.samplesPerPixel = volumeMarch ? 16 : 256;
    }
    TaskSystem tasks;
    settings.tasks = &tasks;
    const auto start = std::chrono::steady_clock::now();
    const Image image = volumeMarch ? pt::RenderVolumeMarchReference(*scene, scene->camera, settings, march)
                                    : pt::RenderReference(*scene, scene->camera, settings);
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
// vdb: what an OpenVDB file holds, and what a volume reads of a grid
// ---------------------------------------------------------------------------
int CommandVdb(Args& args)
{
    std::string path;
    std::optional<std::string> read;
    u32 maxResolution = renderer::Volume{}.maxResolution;
    while (!args.Done())
    {
        const std::string_view arg = args.Take();
        if (arg == "--read")
        {
            read = std::string(args.Next());
        }
        else if (arg == "--max-resolution")
        {
            maxResolution = u32(std::stoul(std::string(args.Next())));
        }
        else if (path.empty() && !arg.starts_with("--"))
        {
            path = arg;
        }
        else
        {
            AIR_LOG(Error, "aircli", "unknown option '{}'", arg);
            return Usage();
        }
    }
    if (args.error || path.empty())
    {
        return Usage();
    }
    std::string error;
    const std::optional<renderer::VdbFileInfo> info = renderer::ReadVdbInfo(path, &error);
    if (!info)
    {
        AIR_LOG(Error, "aircli", "{}", error);
        return kExitCheckFailed;
    }
    auto vec3 = [](const auto* v)
    {
        json::Value out = json::Value::MakeArray();
        for (int i = 0; i < 3; ++i)
        {
            out.Push(double(v[i]));
        }
        return out;
    };
    json::Value out = json::Value::MakeObject();
    out["file"] = path;
    out["version"] = info->version;
    out["library"] = std::format("{}.{}", info->libraryMajor, info->libraryMinor);
    json::Value grids = json::Value::MakeArray();
    for (const renderer::VdbGridInfo& g : info->grids)
    {
        json::Value item = json::Value::MakeObject();
        item["name"] = g.name;
        item["valueType"] = g.valueType;
        item["nodes"] = g.nodes;
        item["halfFloat"] = g.halfFloat;
        item["class"] = g.gridClass;
        if (g.hasBounds)
        {
            item["boundsMin"] = vec3(g.boundsMin);
            item["boundsMax"] = vec3(g.boundsMax);
        }
        item["activeVoxels"] = g.activeVoxels;
        item["map"] = g.map;
        item["voxelSize"] = vec3(g.voxelSize);
        item["translation"] = vec3(g.translation);
        item["readable"] = g.readable;
        if (!g.readable)
        {
            item["why"] = g.why;
        }
        grids.Push(std::move(item));
    }
    out["grids"] = std::move(grids);
    int exit = kExitOk;
    if (read)
    {
        const auto start = std::chrono::steady_clock::now();
        const std::optional<renderer::VolumeGrid> grid = renderer::ReadVdbGrid(path, *read, maxResolution, &error);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        json::Value item = json::Value::MakeObject();
        if (grid)
        {
            double sum = 0.0;
            float largest = 0.0f;
            u64 occupied = 0;
            for (float d : grid->density)
            {
                sum += d;
                largest = std::max(largest, d);
                occupied += d > 0.0f ? 1u : 0u;
            }
            const u32 size[3] = {grid->sizeX, grid->sizeY, grid->sizeZ};
            item["size"] = vec3(size);
            item["voxelSize"] = grid->voxelSize;
            const float origin[3] = {grid->origin.x, grid->origin.y, grid->origin.z};
            item["origin"] = vec3(origin);
            item["meanDensity"] = grid->density.empty() ? 0.0 : sum / double(grid->density.size());
            item["maxDensity"] = largest;
            item["occupiedFraction"] = grid->density.empty() ? 0.0 : double(occupied) / double(grid->density.size());
            item["seconds"] = seconds;
        }
        else
        {
            item["error"] = error;
            exit = kExitCheckFailed;
        }
        out["read"] = std::move(item);
    }
    std::puts(out.Dump().c_str());
    return exit;
}

// ---------------------------------------------------------------------------
// mlp-check
// ---------------------------------------------------------------------------
int CommandMlpCheck(Args& args)
{
    rhi::Backend backend = rhi::Backend::D3D12;
    std::filesystem::path shaderRoot = renderer::FrameRunnerDesc{}.shaderRoot;
    u32 samples = 1024;
    while (!args.Done())
    {
        const std::string_view arg = args.Take();
        if (arg == "--backend")
        {
            const std::string_view b = args.Next();
            backend = b == "d3d12" ? rhi::Backend::D3D12 : b == "vulkan" ? rhi::Backend::Vulkan : rhi::Backend::Null;
        }
        else if (arg == "--shaders")
        {
            shaderRoot = std::string(args.Next());
        }
        else if (arg == "--samples")
        {
            samples = u32(std::stoul(std::string(args.Next())));
        }
        else
        {
            args.error = true;
        }
    }
    if (args.error)
    {
        return Usage();
    }
    auto device = rhi::CreateDevice({.backend = backend, .enableValidation = true});
    if (!device)
    {
        AIR_LOG(Error, "aircli", "could not create a {} device", rhi::ToString(backend));
        return kExitUsage;
    }
    json::Value report = renderer::CheckNeuralNetwork(*device, shaderRoot / "renderer", samples, 1);
    report["adapter"] = device->GetCapabilities().adapterName;
    const bool passed = report["passed"].AsBool();
    std::puts(report.Dump().c_str());
    return passed ? kExitOk : kExitCheckFailed;
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
    SceneOptions sceneOptions;
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
        else if (sceneOptions.Parse(arg, args))
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
        sceneOptions.Apply(*sceneDesc);
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
    if (command == "mlp-check")
    {
        return CommandMlpCheck(args);
    }
    if (command == "vdb")
    {
        return CommandVdb(args);
    }
    return Usage();
}
