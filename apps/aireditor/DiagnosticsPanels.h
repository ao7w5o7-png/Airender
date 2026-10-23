#pragma once

// Profiler and memory windows: GPU pass timeline and table, CPU thread timeline, rolling graphs,
// VRAM budget, allocator pools and heaps, the live resource list and process memory.

#include "Widgets.h"

#include "air/core/Profiler.h"
#include "air/renderer/FrameRunner.h"
#include "air/rhi/RHIDevice.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace air::editor
{

class DiagnosticsPanels
{
public:
    // Feeds one finished frame. `wallMs` is the time between two presented frames.
    void Update(const renderer::FrameResult& frame, double wallMs, u32 samplesThisFrame);

    void BuildProfiler(bool* open);
    void BuildMemory(rhi::Device& device, bool* open);

    // Live values for the menu bar and viewport overlay; they keep updating while paused. GPU time is
    // busy time: waits (for another queue, for a swapchain buffer at vsync) are not counted.
    double GetGpuBusyMs() const { return mLiveGpuMs; }
    float GetFps() const;
    float GetSamplesPerSecond() const { return mLiveSamplesPerSecond; }

    // Paused, the profiler and memory windows keep showing the frame they were paused on (graphs,
    // timelines, tables, resource list) so it can be inspected; StepFrame() captures one more.
    bool IsPaused() const { return mPaused; }
    void SetPaused(bool paused);
    void StepFrame() { mStepFrames = 1; }
    u64 GetCapturedFrame() const { return mCapturedUpdate; } // Update() count of the shown data

    // The profiler's "Export HTML" button: the editor takes the request and reports progress.
    bool TakeReportRequest() { return std::exchange(mReportRequested, false); }
    void SetReportStatus(std::string status) { mReportStatus = std::move(status); }

private:
    struct ScopeStats
    {
        double lastMs = 0.0;
        double averageMs = 0.0; // exponential moving average
        double maxMs = 0.0;     // over the last few seconds
        u32 depth = 0;
        u64 lastSeen = 0;
    };

    void BuildPauseControls();
    void BuildGpuSection();
    void BuildCpuSection();
    void DrawGpuTimeline(float height);
    void DrawCpuTimeline();

    u64 mUpdateCount = 0;
    u64 mCapturedUpdate = 0;
    History mLiveWallMs;
    double mLiveGpuMs = 0.0;
    float mLiveSamplesPerSecond = 0.0f;
    History mWallMs;
    History mCpuMs;
    History mGpuMs;
    History mSamplesPerSecond;
    History mVramUsageMb;
    History mProcessMb;
    rhi::GpuFrameTimings mLastGpu;
    CpuFrame mLastCpu;
    std::vector<rdg::RenderGraph::PassTiming> mPassCpu;
    std::map<std::string, ScopeStats> mGpuScopes;
    bool mPaused = false;
    u32 mStepFrames = 0;
    bool mReportRequested = false;
    std::string mReportStatus;

    // Memory window.
    rhi::MemoryStats mMemory;
    ProcessMemory mProcess;
    std::vector<rhi::ResourceInfo> mResources;
    u64 mResourcesFrame = ~0ull;
    char mResourceFilter[64] = {};
};

} // namespace air::editor
