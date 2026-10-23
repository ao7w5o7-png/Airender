#pragma once

// Profiler and memory windows: GPU pass timeline and table, CPU thread timeline, rolling graphs,
// VRAM budget, allocator pools and heaps, the live resource list and process memory.

#include "Widgets.h"

#include "air/core/Profiler.h"
#include "air/renderer/FrameRunner.h"
#include "air/rhi/RHIDevice.h"

#include <map>
#include <string>
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

    double GetGpuFrameMs() const { return mLastGpu.frameMs; }
    float GetFps() const;
    float GetSamplesPerSecond() const { return mSamplesPerSecond.Last(); }

private:
    struct ScopeStats
    {
        double lastMs = 0.0;
        double averageMs = 0.0; // exponential moving average
        double maxMs = 0.0;     // over the last few seconds
        u32 depth = 0;
        u64 lastSeen = 0;
    };

    void BuildGpuSection();
    void BuildCpuSection();
    void DrawGpuTimeline(float height);
    void DrawCpuTimeline();

    u64 mUpdateCount = 0;
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
    bool mPauseCapture = false;

    // Memory window.
    rhi::MemoryStats mMemory;
    ProcessMemory mProcess;
    std::vector<rhi::ResourceInfo> mResources;
    u64 mResourcesFrame = ~0ull;
    char mResourceFilter[64] = {};
};

} // namespace air::editor
