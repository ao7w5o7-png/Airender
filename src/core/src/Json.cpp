#include "air/core/Json.h"

#include "air/core/Assert.h"

#include <cmath>
#include <format>

namespace air::json {

Value& Value::operator[](const std::string& key) {
    if (std::holds_alternative<std::nullptr_t>(mData)) {
        mData = Object{};
    }
    AIR_CHECK(std::holds_alternative<Object>(mData), "json: operator[] on a non-object value");
    return std::get<Object>(mData)[key];
}

void Value::Push(Value v) {
    if (std::holds_alternative<std::nullptr_t>(mData)) {
        mData = Array{};
    }
    AIR_CHECK(std::holds_alternative<Array>(mData), "json: Push on a non-array value");
    std::get<Array>(mData).push_back(std::move(v));
}

std::string Escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    for (char c : text) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += std::format("\\u{:04x}", int(c));
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string Value::Dump(int indent) const {
    std::string out;
    DumpTo(out, indent, 0);
    return out;
}

void Value::DumpTo(std::string& out, int indent, int depth) const {
    auto newline = [&](int d) {
        if (indent > 0) {
            out += '\n';
            out.append(size_t(indent) * size_t(d), ' ');
        }
    };
    if (std::holds_alternative<std::nullptr_t>(mData)) {
        out += "null";
    } else if (const bool* b = std::get_if<bool>(&mData)) {
        out += *b ? "true" : "false";
    } else if (const double* d = std::get_if<double>(&mData)) {
        if (!std::isfinite(*d)) {
            out += "null";
        } else if (*d == std::floor(*d) && std::fabs(*d) < 1e15) {
            out += std::format("{}", i64(*d));
        } else {
            out += std::format("{}", *d);
        }
    } else if (const std::string* s = std::get_if<std::string>(&mData)) {
        out += '"' + Escape(*s) + '"';
    } else if (const Array* a = std::get_if<Array>(&mData)) {
        out += '[';
        for (size_t i = 0; i < a->size(); ++i) {
            out += i ? "," : "";
            newline(depth + 1);
            (*a)[i].DumpTo(out, indent, depth + 1);
        }
        if (!a->empty()) {
            newline(depth);
        }
        out += ']';
    } else {
        const Object& o = std::get<Object>(mData);
        out += '{';
        bool first = true;
        for (const auto& [key, value] : o) {
            out += first ? "" : ",";
            first = false;
            newline(depth + 1);
            out += '"' + Escape(key) + "\":" + (indent > 0 ? " " : "");
            value.DumpTo(out, indent, depth + 1);
        }
        if (!o.empty()) {
            newline(depth);
        }
        out += '}';
    }
}

} // namespace air::json
