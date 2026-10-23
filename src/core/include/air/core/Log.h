#pragma once

#include <format>
#include <functional>
#include <string_view>

namespace air
{

enum class LogLevel
{
    Trace,
    Info,
    Warning,
    Error,
    Fatal
};

// Thread-safe: lines from different threads never interleave.
void LogMessage(LogLevel level, std::string_view category, std::string_view message);
void SetMinLogLevel(LogLevel level);
const char* ToString(LogLevel level);

// Receives every message that passes the level filter, in addition to stdout/stderr (e.g. an
// editor log window). Called under the log lock: keep it short and never log from it.
using LogSink = std::function<void(LogLevel level, std::string_view category, std::string_view message)>;
void SetLogSink(LogSink sink);

} // namespace air

#define AIR_LOG(level, category, ...) ::air::LogMessage(::air::LogLevel::level, category, std::format(__VA_ARGS__))
