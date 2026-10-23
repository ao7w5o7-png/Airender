#include "air/core/Log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>

namespace air {

namespace {
std::mutex gLogMutex;
std::atomic<LogLevel> gMinLevel{LogLevel::Info};
LogSink gSink; // guarded by gLogMutex

} // namespace

const char* ToString(LogLevel level) {
    switch (level) {
    case LogLevel::Trace: return "trace";
    case LogLevel::Info: return "info";
    case LogLevel::Warning: return "warn";
    case LogLevel::Error: return "error";
    case LogLevel::Fatal: return "fatal";
    }
    return "?";
}

void SetLogSink(LogSink sink) {
    std::lock_guard lock(gLogMutex);
    gSink = std::move(sink);
}

void SetMinLogLevel(LogLevel level) {
    gMinLevel.store(level, std::memory_order_relaxed);
}

void LogMessage(LogLevel level, std::string_view category, std::string_view message) {
    if (level < gMinLevel.load(std::memory_order_relaxed)) {
        return;
    }
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
    std::lock_guard lock(gLogMutex);
    std::FILE* out = level >= LogLevel::Warning ? stderr : stdout;
    std::fprintf(out, "[%s][%s][t%04zu] %.*s\n", ToString(level), std::string(category).c_str(), tid,
                 int(message.size()), message.data());
    std::fflush(out);
    if (gSink) {
        gSink(level, category, message);
    }
}

} // namespace air
