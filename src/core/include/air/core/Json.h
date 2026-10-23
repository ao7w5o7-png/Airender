#pragma once

// Minimal JSON value, serializer and parser. Used for machine readable reports that tools and AI
// agents parse (aircli --report, image comparison results) and for build-generated manifests the
// renderer reads (the shader hot reload manifest).

#include "air/core/Types.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace air::json
{

class Value
{
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

    // Read access. The As* accessors return the fallback (or an empty string / array) when the
    // value holds another type, so missing data reads as empty instead of throwing.
    bool IsNull() const { return std::holds_alternative<std::nullptr_t>(mData); }
    bool IsBool() const { return std::holds_alternative<bool>(mData); }
    bool IsNumber() const { return std::holds_alternative<double>(mData); }
    bool IsString() const { return std::holds_alternative<std::string>(mData); }
    bool IsArray() const { return std::holds_alternative<Array>(mData); }
    bool IsObject() const { return std::holds_alternative<Object>(mData); }
    bool AsBool(bool fallback = false) const;
    double AsNumber(double fallback = 0.0) const;
    const std::string& AsString() const;
    const Array& AsArray() const;
    // Object member, or nullptr when missing or when this is not an object.
    const Value* Find(const std::string& key) const;

private:
    void DumpTo(std::string& out, int indent, int depth) const;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> mData;
};

std::string Escape(std::string_view text);

// Parses JSON text (RFC 8259, UTF-8). On malformed input returns nullopt and, if `error` is given,
// describes the problem and its byte offset.
std::optional<Value> Parse(std::string_view text, std::string* error = nullptr);

} // namespace air::json
