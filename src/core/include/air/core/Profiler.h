#pragma once

// CPU profiler: named scopes recorded per thread and collected per frame.
//
//   AIR_PROFILE_SCOPE("RDG.Compile");          // times the rest of the enclosing block
//   Profiler::Get().SetEnabled(true);
//   Profiler::Get().BeginFrame(); ... Profiler::Get().EndFrame();
//   CpuFrame frame = Profiler::Get().GetLastFrame();   // every scope that finished in that frame
//
// The task system wraps every task in a scope named after the task and the render graph wraps every
// pass, so enabling the profiler is enough to see what each thread did. Cost when disabled: one
// relaxed atomic load per scope.
//
// Threading: all functions are thread-safe. Each thread records into its own buffer; EndFrame()
// collects the buffers under their locks.

#include "air/core/Types.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace air
{

struct CpuEvent
{
    std::string name;
    u32 thread = 0;       // index into CpuFrame::threadNames
    u32 depth = 0;        // nesting level on its thread
    double startMs = 0.0; // relative to the frame start
    double durationMs = 0.0;
};

struct CpuFrame
{
    u64 frameIndex = ~0ull; // ~0: nothing recorded yet
    double durationMs = 0.0;
    std::vector<std::string> threadNames;
    std::vector<CpuEvent> events; // sorted by thread, then start time
};

class Profiler
{
public:
    static Profiler& Get();

    void SetEnabled(bool enabled) { mEnabled.store(enabled, std::memory_order_relaxed); }
    bool IsEnabled() const { return mEnabled.load(std::memory_order_relaxed); }

    // Names the calling thread in the timeline ("Main", "Worker 3", ...).
    void SetThreadName(std::string name);

    void BeginFrame();
    void EndFrame();
    CpuFrame GetLastFrame() const;

    void BeginScope(std::string_view name);
    void EndScope();

    struct ThreadBuffer; // per-thread storage, defined in Profiler.cpp

private:
    Profiler();
    ThreadBuffer& LocalBuffer();
    double NowMs() const;

    std::atomic<bool> mEnabled{false};
    const i64 mEpochNs;
    mutable std::mutex mMutex; // guards everything below
    std::vector<std::shared_ptr<ThreadBuffer>> mThreads;
    double mFrameStartMs = 0.0;
    u64 mFrameIndex = 0;
    CpuFrame mLastFrame;
};

class ProfileScope
{
public:
    explicit ProfileScope(std::string_view name) : mActive(Profiler::Get().IsEnabled())
    {
        if (mActive)
        {
            Profiler::Get().BeginScope(name);
        }
    }
    ~ProfileScope()
    {
        if (mActive)
        {
            Profiler::Get().EndScope();
        }
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    bool mActive;
};

#define AIR_PROFILE_CONCAT_INNER(a, b) a##b
#define AIR_PROFILE_CONCAT(a, b) AIR_PROFILE_CONCAT_INNER(a, b)
#define AIR_PROFILE_SCOPE(name) ::air::ProfileScope AIR_PROFILE_CONCAT(airProfileScope, __LINE__)(name)

// Process memory as the OS reports it (working set / resident set size).
struct ProcessMemory
{
    u64 residentBytes = 0;
    u64 peakResidentBytes = 0;
};
ProcessMemory GetProcessMemory();

} // namespace air
