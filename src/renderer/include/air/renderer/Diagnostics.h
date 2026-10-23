#pragma once

// JSON views of the diagnostic data the engine collects (memory, GPU timings, CPU profile), shared
// by aircli reports, tests and the editor's "copy as JSON" buttons, so humans and AI agents read
// the same numbers the editor plots.

#include "air/core/Json.h"
#include "air/core/Profiler.h"
#include "air/rhi/RHIDevice.h"

namespace air::renderer
{

json::Value ToJson(const rhi::MemoryStats& stats);
json::Value ToJson(const rhi::GpuFrameTimings& timings);
json::Value ToJson(const ProcessMemory& memory);
// Per thread: busy time and the top-level scopes; per scope name: total time and call count.
json::Value ToJson(const CpuFrame& frame);

// "1.50 GiB", "640.0 KiB", ...
std::string FormatBytes(u64 bytes);
const char* ToString(rhi::ResourceInfo::Kind kind);

} // namespace air::renderer
