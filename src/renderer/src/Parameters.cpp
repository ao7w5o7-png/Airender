#include "air/renderer/Parameters.h"

#include "air/core/Assert.h"

#include <charconv>
#include <format>

namespace air::renderer
{

void ParameterSet::Add(std::string name, u32* value, std::string description, u32 min, u32 max)
{
    AIR_CHECK(!Has(name), "parameter '{}' registered twice", name);
    mEntries.push_back({std::move(name), std::move(description), value, double(min), double(max)});
}

void ParameterSet::Add(std::string name, float* value, std::string description, float min, float max)
{
    AIR_CHECK(!Has(name), "parameter '{}' registered twice", name);
    mEntries.push_back({std::move(name), std::move(description), value, double(min), double(max)});
}

void ParameterSet::Add(std::string name, bool* value, std::string description)
{
    AIR_CHECK(!Has(name), "parameter '{}' registered twice", name);
    mEntries.push_back({std::move(name), std::move(description), value, 0.0, 1.0});
}

const ParameterSet::Entry* ParameterSet::Find(std::string_view name) const
{
    for (const Entry& e : mEntries)
    {
        if (e.name == name)
        {
            return &e;
        }
    }
    return nullptr;
}

bool ParameterSet::Has(std::string_view name) const
{
    return Find(name) != nullptr;
}

bool ParameterSet::Set(std::string_view name, std::string_view text, std::string* error)
{
    auto fail = [&](std::string message)
    {
        if (error)
        {
            *error = std::move(message);
        }
        return false;
    };
    const Entry* entry = Find(name);
    if (!entry)
    {
        return fail(std::format("unknown parameter '{}'", name));
    }
    const char* begin = text.data();
    const char* end = text.data() + text.size();

    if (u32* const* p = std::get_if<u32*>(&entry->value))
    {
        u32 v = 0;
        auto [ptr, ec] = std::from_chars(begin, end, v);
        if (ec != std::errc() || ptr != end)
        {
            return fail(std::format("'{}' expects an unsigned integer, got '{}'", name, text));
        }
        if (v < entry->min || v > entry->max)
        {
            return fail(std::format("'{}' must be in [{}, {}], got {}", name, entry->min, entry->max, v));
        }
        **p = v;
    }
    else if (float* const* p = std::get_if<float*>(&entry->value))
    {
        float v = 0;
        auto [ptr, ec] = std::from_chars(begin, end, v);
        if (ec != std::errc() || ptr != end)
        {
            return fail(std::format("'{}' expects a number, got '{}'", name, text));
        }
        if (v < entry->min || v > entry->max)
        {
            return fail(std::format("'{}' must be in [{}, {}], got {}", name, entry->min, entry->max, v));
        }
        **p = v;
    }
    else
    {
        bool* target = std::get<bool*>(entry->value);
        if (text == "1" || text == "true" || text == "on")
        {
            *target = true;
        }
        else if (text == "0" || text == "false" || text == "off")
        {
            *target = false;
        }
        else
        {
            return fail(std::format("'{}' expects true/false, got '{}'", name, text));
        }
    }
    ++mVersion;
    return true;
}

json::Value ParameterSet::ToJson() const
{
    json::Value out = json::Value::MakeObject();
    for (const Entry& e : mEntries)
    {
        json::Value& item = out[e.name];
        item["description"] = e.description;
        if (u32* const* p = std::get_if<u32*>(&e.value))
        {
            item["type"] = "u32";
            item["value"] = **p;
            item["min"] = e.min;
            item["max"] = e.max;
        }
        else if (float* const* p = std::get_if<float*>(&e.value))
        {
            item["type"] = "float";
            item["value"] = **p;
            item["min"] = e.min;
            item["max"] = e.max;
        }
        else
        {
            item["type"] = "bool";
            item["value"] = *std::get<bool*>(e.value);
        }
    }
    return out;
}

} // namespace air::renderer
