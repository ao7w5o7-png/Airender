#include "air/renderer/Diagnostics.h"

#include <algorithm>
#include <format>
#include <map>

namespace air::renderer
{

namespace
{

const char* HeapCategoryName(u32 memoryTypeKey)
{
    // Key encoding shared by the Null and D3D12 backends: location * 4 + category.
    switch (memoryTypeKey % 4)
    {
    case 0: return "Buffers";
    case 1: return "RT/DS textures";
    case 2: return "Textures";
    default: return "Mixed (transient)";
    }
}

} // namespace

std::string FormatBytes(u64 bytes)
{
    static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = double(bytes);
    u32 unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(kUnits))
    {
        value /= 1024.0;
        ++unit;
    }
    return unit == 0 ? std::format("{} B", bytes) : std::format("{:.2f} {}", value, kUnits[unit]);
}

const char* ToString(rhi::ResourceInfo::Kind kind)
{
    switch (kind)
    {
    case rhi::ResourceInfo::Kind::Buffer: return "Buffer";
    case rhi::ResourceInfo::Kind::Texture: return "Texture";
    case rhi::ResourceInfo::Kind::AccelStruct: return "AccelStruct";
    case rhi::ResourceInfo::Kind::Heap: return "Heap";
    }
    return "?";
}

json::Value ToJson(const rhi::MemoryStats& stats)
{
    json::Value out = json::Value::MakeObject();
    json::Value locations = json::Value::MakeObject();
    for (u32 i = 0; i < u32(rhi::MemoryLocation::Count); ++i)
    {
        const auto& s = stats.perLocation[i];
        json::Value& m = locations[rhi::ToString(rhi::MemoryLocation(i))];
        m["heaps"] = s.heapCount;
        m["reservedBytes"] = s.reservedBytes;
        m["usedBytes"] = s.usedBytes;
        m["allocations"] = s.allocationCount;
        m["dedicated"] = s.dedicatedCount;
    }
    out["locations"] = std::move(locations);
    json::Value pools = json::Value::MakeArray();
    for (const auto& pool : stats.pools)
    {
        json::Value p = json::Value::MakeObject();
        p["memoryTypeKey"] = pool.memoryTypeKey;
        p["location"] = rhi::ToString(pool.location);
        p["category"] = HeapCategoryName(pool.memoryTypeKey);
        p["reservedBytes"] = pool.stats.reservedBytes;
        p["usedBytes"] = pool.stats.usedBytes;
        p["allocations"] = pool.stats.allocationCount;
        json::Value heaps = json::Value::MakeArray();
        for (const auto& heap : pool.heaps)
        {
            json::Value h = json::Value::MakeObject();
            h["size"] = heap.size;
            h["used"] = heap.usedBytes;
            h["allocations"] = heap.allocationCount;
            h["dedicated"] = heap.dedicated;
            heaps.Push(std::move(h));
        }
        p["heaps"] = std::move(heaps);
        pools.Push(std::move(p));
    }
    out["pools"] = std::move(pools);
    out["vramBudgetBytes"] = stats.local.budgetBytes;
    out["vramUsageBytes"] = stats.local.usageBytes;
    out["sharedBudgetBytes"] = stats.nonLocal.budgetBytes;
    out["sharedUsageBytes"] = stats.nonLocal.usageBytes;
    out["liveBuffers"] = stats.liveBuffers;
    out["liveTextures"] = stats.liveTextures;
    out["liveAccelStructs"] = stats.liveAccelStructs;
    out["livePipelines"] = stats.livePipelines;
    out["accelStructBytes"] = stats.accelStructBytes;
    out["bindlessDescriptors"] = stats.bindlessDescriptors;
    return out;
}

json::Value ToJson(const rhi::GpuFrameTimings& timings)
{
    json::Value out = json::Value::MakeObject();
    if (timings.frameIndex == ~0ull)
    {
        out["available"] = false;
        return out;
    }
    out["available"] = true;
    out["frame"] = timings.frameIndex;
    out["frameMs"] = timings.frameMs;
    out["busyMs"] = timings.busyMs;
    json::Value scopes = json::Value::MakeArray();
    for (const auto& scope : timings.scopes)
    {
        json::Value s = json::Value::MakeObject();
        s["name"] = scope.name;
        s["queue"] = rhi::ToString(scope.queue);
        s["depth"] = scope.depth;
        s["beginMs"] = scope.beginMs;
        s["ms"] = scope.durationMs;
        scopes.Push(std::move(s));
    }
    out["scopes"] = std::move(scopes);
    return out;
}

json::Value ToJson(const ProcessMemory& memory)
{
    json::Value out = json::Value::MakeObject();
    out["residentBytes"] = memory.residentBytes;
    out["peakResidentBytes"] = memory.peakResidentBytes;
    return out;
}

json::Value ToJson(const CpuFrame& frame)
{
    json::Value out = json::Value::MakeObject();
    if (frame.frameIndex == ~0ull)
    {
        out["available"] = false;
        return out;
    }
    out["available"] = true;
    out["frame"] = frame.frameIndex;
    out["frameMs"] = frame.durationMs;
    json::Value threads = json::Value::MakeArray();
    for (u32 t = 0; t < frame.threadNames.size(); ++t)
    {
        double busy = 0.0;
        u32 scopes = 0;
        for (const CpuEvent& e : frame.events)
        {
            if (e.thread == t && e.depth == 0)
            {
                busy += e.durationMs;
                ++scopes;
            }
        }
        json::Value thread = json::Value::MakeObject();
        thread["name"] = frame.threadNames[t];
        thread["busyMs"] = busy;
        thread["topLevelScopes"] = scopes;
        threads.Push(std::move(thread));
    }
    out["threads"] = std::move(threads);
    std::map<std::string, std::pair<double, u32>> totals;
    for (const CpuEvent& e : frame.events)
    {
        auto& [ms, count] = totals[e.name];
        ms += e.durationMs;
        ++count;
    }
    std::vector<std::pair<std::string, std::pair<double, u32>>> sorted(totals.begin(), totals.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    json::Value scopeTotals = json::Value::MakeArray();
    for (const auto& [name, value] : sorted)
    {
        json::Value s = json::Value::MakeObject();
        s["name"] = name;
        s["totalMs"] = value.first;
        s["count"] = value.second;
        scopeTotals.Push(std::move(s));
    }
    out["scopes"] = std::move(scopeTotals);
    return out;
}

} // namespace air::renderer
