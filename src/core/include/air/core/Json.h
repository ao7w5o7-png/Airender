#pragma once

// Minimal JSON value + serializer (write only). Used for machine readable reports that tools and
// AI agents parse (aircli --report, image comparison results).

#include "air/core/Types.h"

#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace air::json {

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>; // sorted keys keep output deterministic

    Value() : mData(nullptr) {}
    Value(std::nullptr_t) : mData(nullptr) {}
    Value(bool v) : mData(v) {}
    Value(int v) : mData(double(v)) {}
    Value(u32 v) : mData(double(v)) {}
    Value(i64 v) : mData(double(v)) {}
    Value(u64 v) : mData(double(v)) {}
    Value(double v) : mData(v) {}
    Value(float v) : mData(double(v)) {}
    Value(const char* v) : mData(std::string(v)) {}
    Value(std::string v) : mData(std::move(v)) {}
    Value(std::string_view v) : mData(std::string(v)) {}
    Value(Array v) : mData(std::move(v)) {}
    Value(Object v) : mData(std::move(v)) {}

    static Value MakeArray() { return Value(Array{}); }
    static Value MakeObject() { return Value(Object{}); }

    // Object access; converts a null value into an object.
    Value& operator[](const std::string& key);
    // Array append; converts a null value into an array.
    void Push(Value v);

    std::string Dump(int indent = 2) const;

private:
    void DumpTo(std::string& out, int indent, int depth) const;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> mData;
};

std::string Escape(std::string_view text);

} // namespace air::json
