#include "air/renderer/ProfileReport.h"

#include "air/core/Assert.h"
#include "air/renderer/Diagnostics.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <map>
#include <set>

namespace air::embedded
{
std::string_view ProfileReportHtml(); // report/ProfileReport.html, embedded by cmake/AirEmbed.cmake
}

namespace air::renderer
{

namespace
{

// GPU pipeline stages in pipeline order. Units are generic: shader cores are SMs / CUs.
enum Stage : u32
{
    kFrontEnd,
    kWait,
    kSync,
    kCopy,
    kClear,
    kInputAssembly,
    kVertex,
    kRaster,
    kPixel,
    kOutputMerger,
    kComputeShader,
    kRayGen,
    kTraversal,
    kHitMiss,
    kAccelBuild,
    kPresent,
    kHost,
    kStageCount
};

struct StageInfo
{
    std::string_view id;
    std::string_view label;
    std::string_view group;
    std::string_view unit;
    std::string_view note;
};

constexpr StageInfo kStages[kStageCount] = {
    {"frontend", "Front end", "Command processor", "Command processor",
     "Reads the command stream: pipeline and root constant changes, markers, indirect arguments."},
    {"wait", "Queue wait", "Command processor", "Command processor",
     "Stalls this queue until another queue's fence reaches a value."},
    {"sync", "Barrier", "Command processor", "Command processor, caches",
     "Waits for the earlier work using the resource, flushes or invalidates caches, changes the layout."},
    {"copy", "Copy", "Copy", "Copy engine or shader copy", "Moves buffer and texture data."},
    {"clear", "Clear", "Copy", "Clear hardware or a shader", "Fills a texture with one value."},
    {"ia", "Input assembly", "Raster pipeline", "Fixed function", "Fetches indices and vertices."},
    {"vs", "Vertex shader", "Raster pipeline", "Shader cores", "Transforms vertices."},
    {"raster", "Rasterizer", "Raster pipeline", "Fixed function", "Turns triangles into pixel quads."},
    {"ps", "Pixel shader", "Raster pipeline", "Shader cores", "Shades pixels."},
    {"om", "Output merger", "Raster pipeline", "ROPs", "Depth test, blending and render target writes."},
    {"cs", "Compute shader", "Compute", "Shader cores", "The thread groups of a dispatch."},
    {"raygen", "Ray generation", "Ray tracing", "Shader cores", "One thread per launched ray; calls TraceRay."},
    {"traversal", "Traversal", "Ray tracing", "Ray tracing units",
     "BVH traversal and ray / triangle tests in fixed-function hardware."},
    {"hitmiss", "Hit / miss", "Ray tracing", "Shader cores", "Closest hit, any hit and miss shaders."},
    {"asbuild", "AS build", "Ray tracing", "Shader cores, ray tracing units", "Builds or refits a BLAS / TLAS."},
    {"present", "Present", "Display", "Display engine", "Scans out a swapchain buffer."},
    {"host", "CPU read", "Host", "CPU", "Reads mapped memory on the CPU."},
};

constexpr u32 Bit(Stage stage)
{
    return 1u << stage;
}

std::vector<std::string_view> StageIds(u32 mask)
{
    std::vector<std::string_view> ids;
    for (u32 s = 0; s < kStageCount; ++s)
    {
        if (mask & (1u << s))
        {
            ids.push_back(kStages[s].id);
        }
    }
    return ids;
}

json::Value ToJsonArray(const std::vector<std::string_view>& ids)
{
    json::Value out = json::Value::MakeArray();
    for (std::string_view id : ids)
    {
        out.Push(id);
    }
    return out;
}

json::Value StageTable()
{
    json::Value out = json::Value::MakeArray();
    for (const StageInfo& stage : kStages)
    {
        json::Value s = json::Value::MakeObject();
        s["id"] = stage.id;
        s["label"] = stage.label;
        s["group"] = stage.group;
        s["unit"] = stage.unit;
        s["note"] = stage.note;
        out.Push(std::move(s));
    }
    return out;
}

// Stages of every single access state, by name: the page splits "SRVCompute|CopySrc" on '|'.
json::Value AccessStageTable()
{
    json::Value out = json::Value::MakeObject();
    out["None"] = json::Value::MakeArray();
    for (u32 bit = 0; bit < 16; ++bit)
    {
        const rhi::Access access = rhi::Access(1u << bit);
        out[rhi::ToString(access)] = ToJsonArray(AccessStages(access));
    }
    return out;
}

std::string HexId(u64 id)
{
    return std::format("{:#x}", id);
}

json::Value MemoryJson(const FrameCapture& capture)
{
    json::Value out = json::Value::MakeObject();
    out["vramBudgetBytes"] = capture.memory.local.budgetBytes;
    out["vramUsageBytes"] = capture.memory.local.usageBytes;
    out["systemBudgetBytes"] = capture.memory.nonLocal.budgetBytes;
    out["systemUsageBytes"] = capture.memory.nonLocal.usageBytes;

    std::map<u64, std::vector<const rhi::ResourceInfo*>> byHeap;
    for (const rhi::ResourceInfo& r : capture.resources)
    {
        byHeap[r.heapId].push_back(&r);
    }
    auto resourceJson = [](const rhi::ResourceInfo& r)
    {
        json::Value item = json::Value::MakeObject();
        item["name"] = r.name;
        item["kind"] = ToString(r.kind);
        item["size"] = r.sizeBytes;
        item["location"] = rhi::ToString(r.location);
        return item;
    };

    std::set<u64> heapIds;
    json::Value heaps = json::Value::MakeArray();
    for (const auto& pool : capture.memory.pools)
    {
        for (const auto& heap : pool.heaps)
        {
            heapIds.insert(heap.id);
            json::Value h = json::Value::MakeObject();
            h["id"] = HexId(heap.id);
            h["location"] = rhi::ToString(pool.location);
            h["category"] = HeapCategoryName(pool.memoryTypeKey);
            h["memoryTypeKey"] = pool.memoryTypeKey;
            h["size"] = heap.size;
            h["used"] = heap.usedBytes;
            h["allocations"] = heap.allocationCount;
            h["dedicated"] = heap.dedicated;
            json::Value resources = json::Value::MakeArray();
            if (auto it = byHeap.find(heap.id); it != byHeap.end())
            {
                std::vector<const rhi::ResourceInfo*> sorted = it->second;
                std::sort(sorted.begin(), sorted.end(), [](const rhi::ResourceInfo* a, const rhi::ResourceInfo* b)
                          { return a->heapOffset < b->heapOffset; });
                for (const rhi::ResourceInfo* r : sorted)
                {
                    json::Value item = resourceJson(*r);
                    item["offset"] = r->heapOffset;
                    item["placed"] = r->placedInHeap;
                    resources.Push(std::move(item));
                }
            }
            h["resources"] = std::move(resources);
            heaps.Push(std::move(h));
        }
    }
    out["heaps"] = std::move(heaps);

    // Memory the allocator does not manage (swapchain buffers, ...).
    json::Value unplaced = json::Value::MakeArray();
    for (const rhi::ResourceInfo& r : capture.resources)
    {
        if (!heapIds.contains(r.heapId))
        {
            unplaced.Push(resourceJson(r));
        }
    }
    out["unplaced"] = std::move(unplaced);
    return out;
}

} // namespace

std::vector<std::string_view> CommandStages(std::string_view op)
{
    if (op == "Dispatch")
    {
        return StageIds(Bit(kComputeShader));
    }
    if (op == "DispatchRays")
    {
        return StageIds(Bit(kRayGen) | Bit(kTraversal) | Bit(kHitMiss));
    }
    if (op == "BuildBlas" || op == "BuildTlas")
    {
        return StageIds(Bit(kAccelBuild));
    }
    if (op.starts_with("Copy"))
    {
        return StageIds(Bit(kCopy));
    }
    if (op == "ClearTexture")
    {
        return StageIds(Bit(kClear));
    }
    if (op == "Barrier")
    {
        return StageIds(Bit(kSync));
    }
    if (op == "Wait")
    {
        return StageIds(Bit(kWait));
    }
    if (op == "BeginRenderPass")
    {
        // Its draws are recorded on the native command list (the editor's UI), not captured.
        return StageIds(Bit(kInputAssembly) | Bit(kVertex) | Bit(kRaster) | Bit(kPixel) | Bit(kOutputMerger));
    }
    if (op == "EndRenderPass")
    {
        return StageIds(Bit(kOutputMerger));
    }
    return StageIds(Bit(kFrontEnd));
}

std::vector<std::string_view> AccessStages(rhi::Access access)
{
    using rhi::Access;
    constexpr u32 kNonPixelShaders = Bit(kComputeShader) | Bit(kRayGen) | Bit(kHitMiss);
    constexpr u32 kAllShaders = kNonPixelShaders | Bit(kVertex) | Bit(kPixel);
    u32 mask = 0;
    auto add = [&](Access states, u32 stages) { mask |= Any(access & states) ? stages : 0u; };
    add(Access::CopySrc | Access::CopyDst, Bit(kCopy));
    add(Access::VertexOrIndex, Bit(kInputAssembly));
    add(Access::Constant, kAllShaders);
    add(Access::SRVGraphics, Bit(kVertex) | Bit(kPixel));
    add(Access::SRVCompute | Access::UAVCompute, kNonPixelShaders);
    add(Access::UAVGraphics, Bit(kPixel));
    add(Access::RenderTarget | Access::DepthRead | Access::DepthWrite, Bit(kOutputMerger));
    add(Access::IndirectArgs, Bit(kFrontEnd));
    add(Access::AccelStructRead, Bit(kTraversal));
    add(Access::AccelStructWrite, Bit(kAccelBuild));
    add(Access::Present, Bit(kPresent));
    add(Access::HostRead, Bit(kHost));
    return StageIds(mask);
}

json::Value SplitCommandsByPass(const std::vector<std::string>& lines, const json::Value& graph)
{
    // A queue's BeginEvent markers name its live passes in order.
    std::map<std::pair<std::string, std::string>, std::vector<i64>> passesByName; // (queue, name)
    if (const json::Value* passes = graph.Find("passes"))
    {
        for (const json::Value& pass : passes->AsArray())
        {
            if (!pass.Find("culled")->AsBool())
            {
                passesByName[{pass.Find("queue")->AsString(), pass.Find("name")->AsString()}].push_back(
                    i64(pass.Find("index")->AsNumber()));
            }
        }
    }
    std::map<std::pair<std::string, std::string>, size_t> matched;

    struct QueueState
    {
        u32 depth = 0;
        i64 pass = -1;
        std::vector<json::Value> pending; // commands since the queue's last pass
    };
    std::map<std::string, QueueState> queues;
    json::Value out = json::Value::MakeArray();
    auto emit = [&](json::Value command, i64 pass, std::string_view phase)
    {
        command["pass"] = pass;
        command["phase"] = phase;
        out.Push(std::move(command));
    };
    for (const std::string& line : lines)
    {
        const size_t close = line.find("] ");
        if (!line.starts_with('[') || close == std::string::npos)
        {
            continue; // not a command (a validation message)
        }
        const std::string queue = line.substr(1, close - 1);
        const std::string text = line.substr(close + 2);
        const std::string op = text.substr(0, text.find(' '));
        QueueState& state = queues[queue];
        if (op == "BeginEvent")
        {
            if (state.depth++ == 0)
            {
                const size_t first = text.find('\'');
                const size_t last = text.rfind('\'');
                const std::string name = last > first ? text.substr(first + 1, last - first - 1) : std::string();
                const auto it = passesByName.find({queue, name});
                size_t& next = matched[{queue, name}];
                state.pass = it != passesByName.end() && next < it->second.size() ? it->second[next++] : -1;
                for (json::Value& command : state.pending)
                {
                    emit(std::move(command), state.pass, "before");
                }
                state.pending.clear();
            }
            continue;
        }
        if (op == "EndEvent")
        {
            if (state.depth > 0 && --state.depth == 0)
            {
                state.pass = -1;
            }
            continue;
        }
        json::Value command = json::Value::MakeObject();
        command["queue"] = queue;
        command["op"] = op;
        command["text"] = text;
        command["stages"] = ToJsonArray(CommandStages(op));
        if (state.depth > 0)
        {
            emit(std::move(command), state.pass, "in");
        }
        else
        {
            state.pending.push_back(std::move(command));
        }
    }
    for (auto& [queue, state] : queues)
    {
        for (json::Value& command : state.pending)
        {
            emit(std::move(command), -1, "after");
        }
    }
    return out;
}

json::Value MakeProfileReport(const ProfileReportInput& input)
{
    AIR_CHECK(input.capture, "MakeProfileReport: no frame capture (FrameRunner::RequestCapture)");
    const FrameCapture& capture = *input.capture;
    json::Value report = json::Value::MakeObject();
    report["schema"] = 1;
    report["generated"] = std::format("{:%Y-%m-%dT%H:%M:%SZ}", // ISO 8601 in UTC; the page shows local time
                                      std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
    json::Value& device = report["device"];
    device["backend"] = input.backend;
    device["adapter"] = input.adapter;
    json::Value& frame = report["frame"];
    frame["index"] = capture.frameIndex;
    frame["deviceFrame"] = capture.deviceFrame;
    frame["width"] = capture.width;
    frame["height"] = capture.height;
    frame["renderWidth"] = capture.renderWidth;
    frame["renderHeight"] = capture.renderHeight;
    frame["scene"] = input.scene;
    report["features"] = input.features;
    report["graph"] = capture.graph;
    json::Value gpu = ToJson(input.gpu);
    gpu["matchesCapture"] = input.gpu.frameIndex == capture.deviceFrame;
    report["gpu"] = std::move(gpu);
    report["commands"] = SplitCommandsByPass(capture.commands, capture.graph);
    report["stages"] = StageTable();
    report["accessStages"] = AccessStageTable();
    report["memory"] = MemoryJson(capture);
    return report;
}

std::string MakeProfileHtml(const json::Value& report)
{
    // '<' only occurs inside JSON strings, where < means the same character: the data cannot
    // end the <script> element that holds it.
    std::string data;
    for (const char c : report.Dump(0))
    {
        data += c == '<' ? std::string_view("\\u003c") : std::string_view(&c, 1);
    }
    std::string html(embedded::ProfileReportHtml());
    constexpr std::string_view kMarker = "/*AIR_REPORT_DATA*/";
    const size_t at = html.find(kMarker);
    AIR_CHECK(at != std::string::npos, "the profile report template has no {} marker", kMarker);
    html.replace(at, kMarker.size(), data);
    return html;
}

bool WriteProfileHtml(const std::filesystem::path& path, const json::Value& report)
{
    std::error_code error;
    if (path.has_parent_path())
    {
        std::filesystem::create_directories(path.parent_path(), error);
    }
    std::ofstream file(path, std::ios::binary);
    file << MakeProfileHtml(report);
    return bool(file);
}

} // namespace air::renderer
