// The HTML profile report: the render graph's JSON dump, the pipeline stage mapping, the split of the
// command stream per pass, and the page built from a captured frame.

#include "TestHelpers.h"

#include "air/renderer/ProfileReport.h"

#include <algorithm>

using namespace air;
using namespace airtest;

namespace
{

using Stages = std::vector<std::string_view>;

size_t Count(std::string_view text, std::string_view needle)
{
    size_t count = 0;
    for (size_t at = text.find(needle); at != std::string_view::npos; at = text.find(needle, at + 1))
    {
        ++count;
    }
    return count;
}

const json::Value* FindPass(const json::Value& graph, std::string_view name)
{
    for (const json::Value& pass : graph.Find("passes")->AsArray())
    {
        if (pass.Find("name")->AsString() == name)
        {
            return &pass;
        }
    }
    return nullptr;
}

} // namespace

TEST(ProfileReport_MapsCommandsAndStatesToPipelineStages)
{
    EXPECT(renderer::CommandStages("Dispatch") == Stages{"cs"});
    EXPECT(renderer::CommandStages("DispatchRays") == (Stages{"raygen", "traversal", "hitmiss"}));
    EXPECT(renderer::CommandStages("CopyBufferToTexture") == Stages{"copy"});
    EXPECT(renderer::CommandStages("Barrier") == Stages{"sync"});
    EXPECT(renderer::CommandStages("Wait") == Stages{"wait"});
    EXPECT(renderer::CommandStages("SetPipeline") == Stages{"frontend"});
    EXPECT(renderer::AccessStages(rhi::Access::RenderTarget) == Stages{"om"});
    EXPECT(renderer::AccessStages(rhi::Access::AccelStructRead) == Stages{"traversal"});
    EXPECT(renderer::AccessStages(rhi::Access::None).empty());
    // In pipeline order, whatever the order of the state bits.
    EXPECT(renderer::AccessStages(rhi::Access::SRVCompute | rhi::Access::CopySrc) ==
           (Stages{"copy", "cs", "raygen", "hitmiss"}));
}

// Commands between two passes of a queue (waits, barriers) belong to the next pass on that queue;
// what follows a queue's last pass belongs to none.
TEST(ProfileReport_SplitsCommandsByPass)
{
    const std::optional<json::Value> graph = json::Parse(R"({"passes": [
        {"index": 0, "name": "Draw", "culled": false, "queue": "Graphics"},
        {"index": 1, "name": "Culled", "culled": true},
        {"index": 2, "name": "Filter", "culled": false, "queue": "Compute"}]})");
    EXPECT(graph.has_value());
    const std::vector<std::string> lines = {
        "[Graphics] Barrier [tex 'A' None -> UAVCompute]",
        "[Graphics] BeginEvent 'Draw'",
        "[Graphics] Dispatch 8x8x1",
        "[Graphics] EndEvent",
        "[Compute] Wait Graphics fence 2",
        "[Compute] BeginEvent 'Filter'",
        "[Compute] BeginEvent 'Filter.Inner'",
        "[Compute] Dispatch 4x4x1",
        "[Compute] EndEvent",
        "[Compute] EndEvent",
        "[Graphics] Barrier [tex 'A' UAVCompute -> SRVCompute]",
        "VALIDATION: not a command",
    };
    const json::Value commands = renderer::SplitCommandsByPass(lines, *graph);
    const auto& list = commands.AsArray();
    EXPECT(list.size() == 5);
    auto check = [&](size_t i, std::string_view op, double pass, std::string_view phase, std::string_view queue)
    {
        EXPECT(list[i].Find("op")->AsString() == op && list[i].Find("pass")->AsNumber() == pass &&
               list[i].Find("phase")->AsString() == phase && list[i].Find("queue")->AsString() == queue);
    };
    check(0, "Barrier", 0, "before", "Graphics");
    check(1, "Dispatch", 0, "in", "Graphics");
    check(2, "Wait", 2, "before", "Compute");
    check(3, "Dispatch", 2, "in", "Compute"); // inside a nested event: still the pass
    check(4, "Barrier", -1, "after", "Graphics");
    EXPECT(list[3].Find("stages")->AsArray().size() == 1 && list[3].Find("stages")->AsArray()[0].AsString() == "cs");
}

// The graph's JSON carries the schedule the report draws: queues, submissions with the fork and the
// cross-queue waits, barriers, and transient heap offsets.
TEST(RenderGraph_DumpJsonDescribesTheSchedule)
{
    Headless h({"pathtracer", "example"});
    h.runner->RequestCapture();
    const renderer::FrameResult result = h.Frame();
    EXPECT(result.validationErrors.empty());
    EXPECT(result.capture != nullptr);
    const json::Value& graph = result.capture->graph;
    const json::Value* generate = FindPass(graph, "Example.Generate");
    EXPECT(generate && generate->Find("queue")->AsString() == "Compute");
    EXPECT(generate && !generate->Find("barriers")->AsArray().empty());
    const auto& submissions = graph.Find("submissions")->AsArray();
    EXPECT(!submissions.empty() && submissions[0].Find("fork")->AsBool());
    const bool waitsForFork = std::any_of(submissions.begin(), submissions.end(),
                                          [](const json::Value& s)
                                          {
                                              const auto& waits = s.Find("waits")->AsArray();
                                              return s.Find("queue")->AsString() == "Compute" && waits.size() == 1 &&
                                                     waits[0].Find("batch")->AsNumber() == 0;
                                          });
    EXPECT(waitsForFork);
    bool pattern = false;
    for (const json::Value& r : graph.Find("resources")->AsArray())
    {
        if (r.Find("name")->AsString() == "Example.Pattern")
        {
            pattern = r.Find("origin")->AsString() == "transient" && r.Find("heapOffset") &&
                      r.Find("size")->AsNumber() > 0 && r.Find("async")->AsBool();
        }
    }
    EXPECT(pattern);
    EXPECT(graph.Find("stats")->Find("asyncComputePasses")->AsNumber() == 1);
}

TEST(ProfileReport_BuildsThePageFromACapturedFrame)
{
    Headless h({"pathtracer", "example"});
    h.Frame();
    EXPECT(h.Frame().capture == nullptr); // only when requested
    h.runner->RequestCapture();
    const renderer::FrameResult result = h.Frame();
    EXPECT(result.capture != nullptr);
    const json::Value report = renderer::MakeProfileReport({.capture = result.capture.get(),
                                                            .gpu = h.device->GetGpuTimings(),
                                                            .backend = "Null",
                                                            .adapter = "test",
                                                            .scene = "cornell_box",
                                                            .features = h.runner->DescribeFeatures()});
    EXPECT(report.Find("frame")->Find("scene")->AsString() == "cornell_box");
    EXPECT(report.Find("stages")->AsArray().size() >= 15);
    EXPECT(report.Find("accessStages")->Find("RenderTarget")->AsArray()[0].AsString() == "om");

    // Example.Generate's dispatch runs on the compute queue, after its wait for the fork.
    const json::Value* generate = FindPass(*report.Find("graph"), "Example.Generate");
    EXPECT(generate != nullptr);
    const double generateIndex = generate ? generate->Find("index")->AsNumber() : -1.0;
    bool dispatch = false, wait = false;
    for (const json::Value& c : report.Find("commands")->AsArray())
    {
        if (c.Find("pass")->AsNumber() == generateIndex)
        {
            EXPECT(c.Find("queue")->AsString() == "Compute");
            dispatch |= c.Find("op")->AsString() == "Dispatch";
            wait |= c.Find("op")->AsString() == "Wait" && c.Find("phase")->AsString() == "before";
        }
    }
    EXPECT(dispatch && wait);

    // Persistent resources sit in allocator heaps at offsets inside them.
    bool accumulation = false;
    for (const json::Value& heap : report.Find("memory")->Find("heaps")->AsArray())
    {
        for (const json::Value& r : heap.Find("resources")->AsArray())
        {
            if (r.Find("name")->AsString() == "PathTracer.Accumulation")
            {
                accumulation =
                    r.Find("offset")->AsNumber() + r.Find("size")->AsNumber() <= heap.Find("size")->AsNumber();
            }
        }
    }
    EXPECT(accumulation);

    const std::string html = renderer::MakeProfileHtml(report);
    EXPECT(html.find("/*AIR_REPORT_DATA*/") == std::string::npos);
    EXPECT(html.find("\"Example.Generate\"") != std::string::npos);
    const auto path = std::filesystem::temp_directory_path() / "airender_profile_test" / "frame.html";
    EXPECT(renderer::WriteProfileHtml(path, report));
    EXPECT(std::filesystem::file_size(path) == html.size());
}

// Names are data: one that looks like markup cannot end the <script> element holding the report.
TEST(ProfileReport_EscapesMarkupInNames)
{
    json::Value report = json::Value::MakeObject();
    report["name"] = "</script><script>alert(1)</script>";
    const std::string plain = renderer::MakeProfileHtml(json::Value::MakeObject());
    const std::string html = renderer::MakeProfileHtml(report);
    EXPECT(Count(html, "</script>") == Count(plain, "</script>"));
    EXPECT(html.find("\\u003c/script>\\u003cscript>alert(1)") != std::string::npos);
}
