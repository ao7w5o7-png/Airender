#include "air/core/Profiler.h"

#include <algorithm>
#include <chrono>
#include <format>

#if defined(_WIN32)
#include <windows.h>
// windows.h first
#include <psapi.h>
#else
#include <fstream>
#include <unistd.h>
#endif

namespace air {

struct Profiler::ThreadBuffer {
    struct Open {
        std::string name;
        double startMs;
    };
    std::mutex mutex; // guards done and name; stack is only touched by the owning thread
    std::string name;
    std::vector<Open> stack;
    std::vector<CpuEvent> done; // startMs absolute (since the profiler epoch)
};

namespace {
thread_local std::shared_ptr<Profiler::ThreadBuffer> tlsBuffer;

i64 SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

Profiler& Profiler::Get() {
    static Profiler profiler;
    return profiler;
}

Profiler::Profiler() : mEpochNs(SteadyNowNs()) {}

double Profiler::NowMs() const {
    return double(SteadyNowNs() - mEpochNs) / 1e6;
}

Profiler::ThreadBuffer& Profiler::LocalBuffer() {
    if (!tlsBuffer) {
        tlsBuffer = std::make_shared<ThreadBuffer>();
        std::lock_guard lock(mMutex);
        tlsBuffer->name = std::format("Thread {}", mThreads.size());
        mThreads.push_back(tlsBuffer);
    }
    return *tlsBuffer;
}

void Profiler::SetThreadName(std::string name) {
    ThreadBuffer& buffer = LocalBuffer();
    std::lock_guard lock(buffer.mutex);
    buffer.name = std::move(name);
}

void Profiler::BeginScope(std::string_view name) {
    ThreadBuffer& buffer = LocalBuffer();
    buffer.stack.push_back({std::string(name), NowMs()});
}

void Profiler::EndScope() {
    ThreadBuffer& buffer = LocalBuffer();
    if (buffer.stack.empty()) {
        return; // profiling was enabled inside the scope
    }
    ThreadBuffer::Open open = std::move(buffer.stack.back());
    buffer.stack.pop_back();
    const double end = NowMs();
    std::lock_guard lock(buffer.mutex);
    buffer.done.push_back({std::move(open.name), 0, u32(buffer.stack.size()), open.startMs, end - open.startMs});
}

void Profiler::BeginFrame() {
    std::lock_guard lock(mMutex);
    mFrameStartMs = NowMs();
}

void Profiler::EndFrame() {
    const double end = NowMs();
    CpuFrame frame;
    std::lock_guard lock(mMutex);
    frame.frameIndex = mFrameIndex++;
    frame.durationMs = end - mFrameStartMs;
    // Threads that exited and recorded nothing are dropped from the list.
    std::erase_if(mThreads, [](const std::shared_ptr<ThreadBuffer>& t) {
        std::lock_guard bufferLock(t->mutex);
        return t.use_count() == 1 && t->done.empty();
    });
    for (u32 i = 0; i < mThreads.size(); ++i) {
        ThreadBuffer& buffer = *mThreads[i];
        std::lock_guard bufferLock(buffer.mutex);
        frame.threadNames.push_back(buffer.name);
        for (CpuEvent& event : buffer.done) {
            event.thread = i;
            event.startMs -= mFrameStartMs;
            frame.events.push_back(std::move(event));
        }
        buffer.done.clear();
    }
    std::stable_sort(frame.events.begin(), frame.events.end(), [](const CpuEvent& a, const CpuEvent& b) {
        return a.thread != b.thread ? a.thread < b.thread : a.startMs < b.startMs;
    });
    mFrameStartMs = end;
    mLastFrame = std::move(frame);
}

CpuFrame Profiler::GetLastFrame() const {
    std::lock_guard lock(mMutex);
    return mLastFrame;
}

ProcessMemory GetProcessMemory() {
    ProcessMemory memory;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters = {};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        memory.residentBytes = counters.WorkingSetSize;
        memory.peakResidentBytes = counters.PeakWorkingSetSize;
    }
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        u64 kib = 0;
        if (line.rfind("VmRSS:", 0) == 0 && std::sscanf(line.c_str(), "VmRSS: %llu", (unsigned long long*)&kib) == 1) {
            memory.residentBytes = kib * 1024;
        } else if (line.rfind("VmHWM:", 0) == 0 &&
                   std::sscanf(line.c_str(), "VmHWM: %llu", (unsigned long long*)&kib) == 1) {
            memory.peakResidentBytes = kib * 1024;
        }
    }
#endif
    return memory;
}

} // namespace air
