#pragma once

// Profile report of one frame, as data and as a self-contained HTML page (no network access). The
// page shows:
//   * the GPU timeline, one lane per queue: async compute overlap, the submissions and the
//     cross-queue waits between them,
//   * where each pass's commands run in the GPU pipeline (front end, copy, raster stages, compute,
//     ray tracing units), and what each barrier waits for and unblocks,
//   * the memory: transient resources over the frame's passes (aliasing in the transient heap) and
//     every allocator heap with its resources at their offsets.
// aircli render --html and the editor's Profiler > Export HTML write it (docs/architecture.md §9.2).
//
//   runner.RequestCapture();
//   FrameResult result = runner.RenderFrame(scene, camera);
//   device.WaitIdle(); // or keep rendering until GpuFrameTimings::frameIndex == capture->deviceFrame
//   WriteProfileHtml("out/profile.html", MakeProfileReport({.capture = result.capture.get(), .gpu = ...}));

#include "air/core/Json.h"
#include "air/renderer/FrameRunner.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace air::renderer
{

struct ProfileReportInput
{
    const FrameCapture* capture = nullptr;
    rhi::GpuFrameTimings gpu; // of the captured frame: GpuFrameTimings::frameIndex == capture->deviceFrame
    std::string backend;
    std::string adapter;
    std::string scene;
    json::Value features; // FrameRunner::DescribeFeatures()
};

// The report's data: the inputs plus the command stream split per pass with the pipeline stages of
// every command, and the memory map. The HTML page embeds exactly this.
json::Value MakeProfileReport(const ProfileReportInput& input);
std::string MakeProfileHtml(const json::Value& report);
bool WriteProfileHtml(const std::filesystem::path& path, const json::Value& report);

// GPU pipeline stage ids ("cs", "traversal", "om", ...; the report's "stages" table describes them)
// a captured command runs on, e.g. "DispatchRays" -> raygen, traversal, hitmiss; and the stages an
// access state is used by, e.g. RenderTarget -> om: a barrier waits for its `before` stages and
// unblocks its `after` stages.
std::vector<std::string_view> CommandStages(std::string_view op);
std::vector<std::string_view> AccessStages(rhi::Access access);

// The captured command lines ("[Compute] Dispatch 8x4x1") of a frame with the pass each belongs to,
// per the graph's passes (DumpJson): commands between two passes of a queue (waits, barriers) belong
// to the next one ("before"), the queue's last ones to none ("after").
json::Value SplitCommandsByPass(const std::vector<std::string>& lines, const json::Value& graph);

} // namespace air::renderer
