#pragma once

// Named, typed, documented parameters a feature exposes to the outside world (CLI, reports, AI
// agents, future UI). A feature binds its own fields; callers set them by name from text:
//   params.Add("maxBounces", &mMaxBounces, "Path length limit", 1, 64);
//   runner.SetParameter("pathtracer.maxBounces", "4");
//
// Threading: only modify between frames, on the thread that drives the frame loop.

#include "air/core/Json.h"
#include "air/core/Types.h"

#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace air::renderer
{

class ParameterSet
{
public:
    void Add(std::string name, u32* value, std::string description, u32 min = 0, u32 max = ~0u);
    void Add(std::string name, float* value, std::string description, float min = -1e30f, float max = 1e30f);
    void Add(std::string name, bool* value, std::string description);

    // Parses `text` and assigns it. Returns false and fills `error` on unknown name / bad value / out of range.
    bool Set(std::string_view name, std::string_view text, std::string* error = nullptr);
    bool Has(std::string_view name) const;

    // Incremented on every successful Set(); features compare it to react to changes.
    u64 GetVersion() const { return mVersion; }

    // {"name": {"value": ..., "type": "u32", "min": ..., "max": ..., "description": ...}}
    json::Value ToJson() const;

    struct Entry
    {
        std::string name;
        std::string description;
        std::variant<u32*, float*, bool*> value;
        double min = 0;
        double max = 0;
    };
    // For UIs that edit the bound fields directly: call Touch() after changing one.
    const std::vector<Entry>& Entries() const { return mEntries; }
    void Touch() { ++mVersion; }

private:
    const Entry* Find(std::string_view name) const;

    std::vector<Entry> mEntries;
    u64 mVersion = 0;
};

} // namespace air::renderer
