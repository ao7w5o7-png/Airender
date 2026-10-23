#include "DiagnosticsPanels.h"

#include "air/renderer/Diagnostics.h"

#include <algorithm>
#include <format>

namespace air::editor
{

namespace
{

constexpr float kGraphHeight = 56.0f;
constexpr double kMb = 1024.0 * 1024.0;

const char* KindName(rhi::ResourceInfo::Kind kind)
{
    return renderer::ToString(kind);
}

// Draws one bar of a timeline with a label clipped to it and a tooltip.
void TimelineBar(ImDrawList* draw, ImVec2 min, ImVec2 max, const std::string& name, double ms)
{
    const ImU32 color = ColorForName(name);
    draw->AddRectFilled(min, max, color, 2.0f);
    draw->AddRect(min, max, IM_COL32(0, 0, 0, 90), 2.0f);
    const std::string label = std::format("{} {}", name, FormatMs(ms));
    const ImVec2 size = ImGui::CalcTextSize(label.c_str());
    if (size.x + 6.0f < max.x - min.x)
    {
        draw->AddText(ImVec2(min.x + 3.0f, min.y + (max.y - min.y - size.y) * 0.5f), IM_COL32(15, 15, 15, 255),
                      label.c_str());
    }
    if (ImGui::IsMouseHoveringRect(min, max))
    {
        ImGui::SetTooltip("%s", label.c_str());
    }
}

} // namespace

void DiagnosticsPanels::Update(const renderer::FrameResult& frame, double wallMs, u32 samplesThisFrame)
{
    ++mUpdateCount;
    mWallMs.Push(float(wallMs));
    mCpuMs.Push(float(frame.cpuMilliseconds));
    if (wallMs > 0.0)
    {
        mSamplesPerSecond.Push(float(double(samplesThisFrame) * 1000.0 / wallMs));
    }
    if (mPauseCapture)
    {
        return;
    }
    if (frame.gpu.frameIndex != ~0ull)
    {
        mLastGpu = frame.gpu;
        mGpuMs.Push(float(frame.gpu.frameMs));
        for (const rhi::GpuTimingScope& scope : frame.gpu.scopes)
        {
            ScopeStats& s = mGpuScopes[scope.name];
            s.lastMs = scope.durationMs;
            s.averageMs = s.lastSeen == 0 ? scope.durationMs : s.averageMs * 0.95 + scope.durationMs * 0.05;
            s.maxMs = (mUpdateCount % 240 == 0) ? scope.durationMs : std::max(s.maxMs, scope.durationMs);
            s.depth = scope.depth;
            s.lastSeen = mUpdateCount;
        }
    }
    mPassCpu = frame.passCpu;
    mLastCpu = Profiler::Get().GetLastFrame();
    mMemory = frame.memory;
    mProcess = frame.processMemory;
    mVramUsageMb.Push(float(double(mMemory.local.usageBytes) / kMb));
    mProcessMb.Push(float(double(mProcess.residentBytes) / kMb));
}

float DiagnosticsPanels::GetFps() const
{
    const float ms = mWallMs.Average();
    return ms > 0.0f ? 1000.0f / ms : 0.0f;
}

// --- Profiler ------------------------------------------------------------------------------------

void DiagnosticsPanels::BuildProfiler(bool* open)
{
    if (!ImGui::Begin("Profiler", open))
    {
        ImGui::End();
        return;
    }
    ImGui::Checkbox("Pause", &mPauseCapture);
    ImGui::SameLine();
    ImGui::Text("%.1f fps   frame %s   GPU %s   CPU record %s", GetFps(), FormatMs(mWallMs.Last()).c_str(),
                FormatMs(mLastGpu.frameMs).c_str(), FormatMs(mCpuMs.Last()).c_str());
    PlotHistory("Frame", mWallMs, "ms", kGraphHeight);
    PlotHistory("GPU", mGpuMs, "ms", kGraphHeight);
    PlotHistory("CPU", mCpuMs, "ms", kGraphHeight);
    PlotHistory("Samples/s", mSamplesPerSecond, "spp/s", kGraphHeight);

    if (ImGui::BeginTabBar("profiler"))
    {
        if (ImGui::BeginTabItem("GPU"))
        {
            BuildGpuSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("CPU"))
        {
            BuildCpuSection();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void DiagnosticsPanels::BuildGpuSection()
{
    if (mLastGpu.frameIndex == ~0ull)
    {
        ImGui::TextDisabled("No GPU timings yet (the GPU reports a few frames late).");
        return;
    }
    ImGui::Text("GPU frame %llu: %s", static_cast<unsigned long long>(mLastGpu.frameIndex),
                FormatMs(mLastGpu.frameMs).c_str());
    DrawGpuTimeline(22.0f);
    if (ImGui::BeginTable("gpuScopes", 5,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY,
                          ImVec2(0.0f, 0.0f)))
    {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Last");
        ImGui::TableSetupColumn("Average");
        ImGui::TableSetupColumn("Max");
        ImGui::TableSetupColumn("Share");
        ImGui::TableHeadersRow();
        for (const rhi::GpuTimingScope& scope : mLastGpu.scopes)
        {
            const ScopeStats& s = mGpuScopes[scope.name];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Indent(float(scope.depth) * 12.0f);
            ImGui::ColorButton("##c", ImGui::ColorConvertU32ToFloat4(ColorForName(scope.name)),
                               ImGuiColorEditFlags_NoTooltip, ImVec2(10, 10));
            ImGui::SameLine();
            ImGui::TextUnformatted(scope.name.c_str());
            ImGui::Unindent(float(scope.depth) * 12.0f);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FormatMs(s.lastMs).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FormatMs(s.averageMs).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FormatMs(s.maxMs).c_str());
            ImGui::TableNextColumn();
            const double share = mLastGpu.frameMs > 0.0 ? s.lastMs / mLastGpu.frameMs : 0.0;
            UsageBar(float(share), std::format("{:.0f}%", share * 100.0).c_str(), ColorForName(scope.name));
        }
        ImGui::EndTable();
    }
}

void DiagnosticsPanels::DrawGpuTimeline(float rowHeight)
{
    u32 maxDepth = 0;
    for (const auto& scope : mLastGpu.scopes)
    {
        maxDepth = std::max(maxDepth, scope.depth);
    }
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(ImGui::GetContentRegionAvail().x, 50.0f);
    const float height = float(maxDepth + 1) * (rowHeight + 2.0f);
    ImGui::InvisibleButton("gpuTimeline", ImVec2(width, height));
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), IM_COL32(30, 30, 34, 255));
    const double scale = mLastGpu.frameMs > 0.0 ? width / mLastGpu.frameMs : 0.0;
    for (const rhi::GpuTimingScope& scope : mLastGpu.scopes)
    {
        const float x0 = origin.x + float(scope.beginMs * scale);
        const float x1 = std::max(x0 + 1.0f, origin.x + float((scope.beginMs + scope.durationMs) * scale));
        const float y0 = origin.y + float(scope.depth) * (rowHeight + 2.0f);
        TimelineBar(draw, ImVec2(x0, y0), ImVec2(x1, y0 + rowHeight), scope.name, scope.durationMs);
    }
}

void DiagnosticsPanels::BuildCpuSection()
{
    if (mLastCpu.frameIndex == ~0ull)
    {
        ImGui::TextDisabled("CPU profiler has not finished a frame yet.");
        return;
    }
    ImGui::Text("CPU frame %llu: %s on %zu threads", static_cast<unsigned long long>(mLastCpu.frameIndex),
                FormatMs(mLastCpu.durationMs).c_str(), mLastCpu.threadNames.size());
    DrawCpuTimeline();
    if (ImGui::TreeNodeEx("Render graph passes (recording time)", ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (const auto& pass : mPassCpu)
        {
            ImGui::BulletText("%s  %s", pass.name.c_str(), FormatMs(pass.cpuMs).c_str());
        }
        ImGui::TreePop();
    }
}

void DiagnosticsPanels::DrawCpuTimeline()
{
    constexpr float kRow = 18.0f;
    constexpr float kLabel = 90.0f;
    std::vector<u32> depthPerThread(mLastCpu.threadNames.size(), 0);
    for (const CpuEvent& e : mLastCpu.events)
    {
        depthPerThread[e.thread] = std::max(depthPerThread[e.thread], e.depth + 1);
    }
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(ImGui::GetContentRegionAvail().x - kLabel, 50.0f);
    float height = 0.0f;
    for (u32 depth : depthPerThread)
    {
        height += float(std::max(depth, 1u)) * kRow + 4.0f;
    }
    ImGui::InvisibleButton("cpuTimeline", ImVec2(width + kLabel, height));
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const double scale = mLastCpu.durationMs > 0.0 ? width / mLastCpu.durationMs : 0.0;
    float laneY = origin.y;
    std::vector<float> laneTop(mLastCpu.threadNames.size(), 0.0f);
    for (size_t t = 0; t < mLastCpu.threadNames.size(); ++t)
    {
        const float laneHeight = float(std::max(depthPerThread[t], 1u)) * kRow;
        laneTop[t] = laneY;
        draw->AddRectFilled(ImVec2(origin.x + kLabel, laneY), ImVec2(origin.x + kLabel + width, laneY + laneHeight),
                            t % 2 ? IM_COL32(34, 34, 38, 255) : IM_COL32(28, 28, 32, 255));
        draw->AddText(ImVec2(origin.x, laneY + 2.0f), ImGui::GetColorU32(ImGuiCol_Text),
                      mLastCpu.threadNames[t].c_str());
        laneY += laneHeight + 4.0f;
    }
    for (const CpuEvent& e : mLastCpu.events)
    {
        const float x0 = origin.x + kLabel + float(e.startMs * scale);
        const float x1 = std::max(x0 + 1.0f, origin.x + kLabel + float((e.startMs + e.durationMs) * scale));
        const float y0 = laneTop[e.thread] + float(e.depth) * kRow;
        TimelineBar(draw, ImVec2(x0, y0 + 1.0f), ImVec2(x1, y0 + kRow - 1.0f), e.name, e.durationMs);
    }
}

// --- Memory --------------------------------------------------------------------------------------

void DiagnosticsPanels::BuildMemory(rhi::Device& device, bool* open)
{
    if (!ImGui::Begin("Memory", open))
    {
        ImGui::End();
        return;
    }
    const rhi::MemoryStats& m = mMemory;

    ImGui::SeparatorText("Video memory (OS budget)");
    auto budgetBar = [](const char* label, const rhi::MemoryBudget& b)
    {
        const float fraction = b.budgetBytes ? float(double(b.usageBytes) / double(b.budgetBytes)) : 0.0f;
        const std::string text = std::format("{}: {} / {}", label, renderer::FormatBytes(b.usageBytes),
                                             renderer::FormatBytes(b.budgetBytes));
        UsageBar(fraction, text.c_str(),
                 fraction > 0.9f   ? IM_COL32(220, 70, 60, 255)
                 : fraction > 0.7f ? IM_COL32(220, 170, 50, 255)
                                   : 0);
    };
    budgetBar("Local (VRAM)", m.local);
    budgetBar("Non-local (system)", m.nonLocal);
    PlotHistory("VRAM usage", mVramUsageMb, "MiB", kGraphHeight);

    ImGui::SeparatorText("Process");
    ImGui::Text("Resident %s, peak %s", renderer::FormatBytes(mProcess.residentBytes).c_str(),
                renderer::FormatBytes(mProcess.peakResidentBytes).c_str());
    PlotHistory("Process memory", mProcessMb, "MiB", kGraphHeight);

    ImGui::SeparatorText("Objects");
    ImGui::Text("Buffers %u   Textures %u   Accel structs %u (%s)   Pipelines %u   Bindless slots %u", m.liveBuffers,
                m.liveTextures, m.liveAccelStructs, renderer::FormatBytes(m.accelStructBytes).c_str(), m.livePipelines,
                m.bindlessDescriptors);

    ImGui::SeparatorText("Allocator (by memory location)");
    for (u32 i = 0; i < u32(rhi::MemoryLocation::Count); ++i)
    {
        const auto& s = m.perLocation[i];
        if (s.heapCount == 0 && s.allocationCount == 0)
        {
            continue;
        }
        const float fraction = s.reservedBytes ? float(double(s.usedBytes) / double(s.reservedBytes)) : 0.0f;
        const std::string text = std::format("{}: {} used of {} in {} heaps, {} allocations",
                                             rhi::ToString(rhi::MemoryLocation(i)), renderer::FormatBytes(s.usedBytes),
                                             renderer::FormatBytes(s.reservedBytes), s.heapCount, s.allocationCount);
        UsageBar(fraction, text.c_str());
    }
    if (ImGui::TreeNode("Pools and heaps"))
    {
        for (const auto& pool : m.pools)
        {
            const std::string title =
                std::format("type {} ({}): {} / {}###pool{}", pool.memoryTypeKey, rhi::ToString(pool.location),
                            renderer::FormatBytes(pool.stats.usedBytes),
                            renderer::FormatBytes(pool.stats.reservedBytes), pool.memoryTypeKey);
            if (ImGui::TreeNode(title.c_str()))
            {
                for (size_t h = 0; h < pool.heaps.size(); ++h)
                {
                    const auto& heap = pool.heaps[h];
                    const std::string text = std::format(
                        "heap {}{}: {} / {}, {} allocations", h, heap.dedicated ? " (dedicated)" : "",
                        renderer::FormatBytes(heap.usedBytes), renderer::FormatBytes(heap.size), heap.allocationCount);
                    UsageBar(heap.size ? float(double(heap.usedBytes) / double(heap.size)) : 0.0f, text.c_str());
                }
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }

    ImGui::SeparatorText("Resources");
    // The list walks every live object: refresh twice a second, not every frame.
    if (mResourcesFrame == ~0ull || mUpdateCount - mResourcesFrame > 30 || ImGui::Button("Refresh"))
    {
        mResources = device.GetResourceList();
        std::sort(mResources.begin(), mResources.end(),
                  [](const rhi::ResourceInfo& a, const rhi::ResourceInfo& b) { return a.sizeBytes > b.sizeBytes; });
        mResourcesFrame = mUpdateCount;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200.0f);
    ImGui::InputTextWithHint("##filter", "filter by name", mResourceFilter, sizeof(mResourceFilter));
    u64 total = 0;
    for (const auto& r : mResources)
    {
        total += r.sizeBytes;
    }
    ImGui::SameLine();
    ImGui::Text("%zu resources, %s", mResources.size(), renderer::FormatBytes(total).c_str());
    if (ImGui::BeginTable("resources", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_BordersInnerV))
    {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Kind");
        ImGui::TableSetupColumn("Memory");
        ImGui::TableSetupColumn("Size");
        ImGui::TableHeadersRow();
        const std::string_view filter = mResourceFilter;
        for (const rhi::ResourceInfo& r : mResources)
        {
            if (!filter.empty() && r.name.find(filter) == std::string::npos)
            {
                continue;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(KindName(r.kind));
            ImGui::TableNextColumn();
            ImGui::Text("%s%s", rhi::ToString(r.location), r.placedInHeap ? " (transient heap)" : "");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(renderer::FormatBytes(r.sizeBytes).c_str());
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

} // namespace air::editor
