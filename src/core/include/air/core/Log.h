#pragma once

#include <format>
#include <string_view>

namespace air {

enum class LogLevel { Trace, Info, Warning, Error, Fatal };

// Thread-safe: lines from different threads never interleave.
void LogMessage(LogLevel level, std::string_view category, std::string_view message);
void SetMinLogLevel(LogLevel level);

} // namespace air

#define AIR_LOG(level, category, ...) ::air::LogMessage(::air::LogLevel::level, category, std::format(__VA_ARGS__))
