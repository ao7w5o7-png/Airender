#include "air/core/Json.h"

#include "air/core/Assert.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <format>

namespace air::json
{

Value& Value::operator[](const std::string& key)
{
    if (std::holds_alternative<std::nullptr_t>(mData))
    {
        mData = Object{};
    }
    AIR_CHECK(std::holds_alternative<Object>(mData), "json: operator[] on a non-object value");
    return std::get<Object>(mData)[key];
}

void Value::Push(Value v)
{
    if (std::holds_alternative<std::nullptr_t>(mData))
    {
        mData = Array{};
    }
    AIR_CHECK(std::holds_alternative<Array>(mData), "json: Push on a non-array value");
    std::get<Array>(mData).push_back(std::move(v));
}

std::string Escape(std::string_view text)
{
    std::string out;
    out.reserve(text.size() + 2);
    for (char c : text)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                out += std::format("\\u{:04x}", int(c));
            }
            else
            {
                out += c;
            }
        }
    }
    return out;
}

std::string Value::Dump(int indent) const
{
    std::string out;
    DumpTo(out, indent, 0);
    return out;
}

void Value::DumpTo(std::string& out, int indent, int depth) const
{
    auto newline = [&](int d)
    {
        if (indent > 0)
        {
            out += '\n';
            out.append(size_t(indent) * size_t(d), ' ');
        }
    };
    if (std::holds_alternative<std::nullptr_t>(mData))
    {
        out += "null";
    }
    else if (const bool* b = std::get_if<bool>(&mData))
    {
        out += *b ? "true" : "false";
    }
    else if (const double* d = std::get_if<double>(&mData))
    {
        if (!std::isfinite(*d))
        {
            out += "null";
        }
        else if (*d == std::floor(*d) && std::fabs(*d) < 1e15)
        {
            out += std::format("{}", i64(*d));
        }
        else
        {
            out += std::format("{}", *d);
        }
    }
    else if (const std::string* s = std::get_if<std::string>(&mData))
    {
        out += '"' + Escape(*s) + '"';
    }
    else if (const Array* a = std::get_if<Array>(&mData))
    {
        out += '[';
        for (size_t i = 0; i < a->size(); ++i)
        {
            out += i ? "," : "";
            newline(depth + 1);
            (*a)[i].DumpTo(out, indent, depth + 1);
        }
        if (!a->empty())
        {
            newline(depth);
        }
        out += ']';
    }
    else
    {
        const Object& o = std::get<Object>(mData);
        out += '{';
        bool first = true;
        for (const auto& [key, value] : o)
        {
            out += first ? "" : ",";
            first = false;
            newline(depth + 1);
            out += '"' + Escape(key) + "\":" + (indent > 0 ? " " : "");
            value.DumpTo(out, indent, depth + 1);
        }
        if (!o.empty())
        {
            newline(depth);
        }
        out += '}';
    }
}

bool Value::AsBool(bool fallback) const
{
    const bool* b = std::get_if<bool>(&mData);
    return b ? *b : fallback;
}

double Value::AsNumber(double fallback) const
{
    const double* d = std::get_if<double>(&mData);
    return d ? *d : fallback;
}

const std::string& Value::AsString() const
{
    static const std::string kEmpty;
    const std::string* s = std::get_if<std::string>(&mData);
    return s ? *s : kEmpty;
}

const Value::Array& Value::AsArray() const
{
    static const Array kEmpty;
    const Array* a = std::get_if<Array>(&mData);
    return a ? *a : kEmpty;
}

const Value* Value::Find(const std::string& key) const
{
    const Object* o = std::get_if<Object>(&mData);
    if (!o)
    {
        return nullptr;
    }
    const auto it = o->find(key);
    return it == o->end() ? nullptr : &it->second;
}

namespace
{

// Recursive descent over the RFC 8259 grammar. Nesting is limited so hostile input cannot overflow
// the stack.
class Parser
{
public:
    explicit Parser(std::string_view text) : mText(text) {}

    std::optional<Value> Document()
    {
        std::optional<Value> value = ParseValue(0);
        SkipWhitespace();
        if (value && mPos != mText.size())
        {
            return Fail("unexpected trailing characters");
        }
        return value;
    }

    const std::string& Error() const { return mError; }

private:
    static constexpr int kMaxDepth = 256;

    std::nullopt_t Fail(std::string_view what)
    {
        if (mError.empty())
        {
            mError = std::format("{} at byte {}", what, mPos);
        }
        return std::nullopt;
    }

    void SkipWhitespace()
    {
        while (mPos < mText.size() &&
               (mText[mPos] == ' ' || mText[mPos] == '\t' || mText[mPos] == '\n' || mText[mPos] == '\r'))
        {
            ++mPos;
        }
    }

    bool Consume(std::string_view token)
    {
        if (mText.substr(mPos, token.size()) == token)
        {
            mPos += token.size();
            return true;
        }
        return false;
    }

    std::optional<Value> ParseValue(int depth)
    {
        if (depth > kMaxDepth)
        {
            return Fail("nesting too deep");
        }
        SkipWhitespace();
        if (mPos >= mText.size())
        {
            return Fail("unexpected end of input");
        }
        const char c = mText[mPos];
        if (c == '{')
        {
            return ParseObject(depth);
        }
        if (c == '[')
        {
            return ParseArray(depth);
        }
        if (c == '"')
        {
            std::optional<std::string> s = ParseString();
            return s ? std::optional<Value>(Value(std::move(*s))) : std::nullopt;
        }
        if (Consume("true"))
        {
            return Value(true);
        }
        if (Consume("false"))
        {
            return Value(false);
        }
        if (Consume("null"))
        {
            return Value(nullptr);
        }
        return ParseNumber();
    }

    std::optional<Value> ParseObject(int depth)
    {
        ++mPos; // {
        Value object = Value::MakeObject();
        SkipWhitespace();
        if (Consume("}"))
        {
            return object;
        }
        for (;;)
        {
            SkipWhitespace();
            if (mPos >= mText.size() || mText[mPos] != '"')
            {
                return Fail("expected a member name");
            }
            std::optional<std::string> key = ParseString();
            if (!key)
            {
                return std::nullopt;
            }
            SkipWhitespace();
            if (!Consume(":"))
            {
                return Fail("expected ':'");
            }
            std::optional<Value> value = ParseValue(depth + 1);
            if (!value)
            {
                return std::nullopt;
            }
            object[*key] = std::move(*value);
            SkipWhitespace();
            if (Consume("}"))
            {
                return object;
            }
            if (!Consume(","))
            {
                return Fail("expected ',' or '}'");
            }
        }
    }

    std::optional<Value> ParseArray(int depth)
    {
        ++mPos; // [
        Value array = Value::MakeArray();
        SkipWhitespace();
        if (Consume("]"))
        {
            return array;
        }
        for (;;)
        {
            std::optional<Value> value = ParseValue(depth + 1);
            if (!value)
            {
                return std::nullopt;
            }
            array.Push(std::move(*value));
            SkipWhitespace();
            if (Consume("]"))
            {
                return array;
            }
            if (!Consume(","))
            {
                return Fail("expected ',' or ']'");
            }
        }
    }

    std::optional<u32> ParseHex4()
    {
        if (mPos + 4 > mText.size())
        {
            return Fail("truncated \\u escape");
        }
        u32 value = 0;
        const auto [end, ec] = std::from_chars(mText.data() + mPos, mText.data() + mPos + 4, value, 16);
        if (ec != std::errc() || end != mText.data() + mPos + 4)
        {
            return Fail("invalid \\u escape");
        }
        mPos += 4;
        return value;
    }

    static void AppendUtf8(std::string& out, u32 cp)
    {
        if (cp < 0x80)
        {
            out += char(cp);
        }
        else if (cp < 0x800)
        {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        }
        else if (cp < 0x10000)
        {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
        else
        {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    std::optional<std::string> ParseString()
    {
        ++mPos; // opening quote
        std::string out;
        while (mPos < mText.size())
        {
            const char c = mText[mPos++];
            if (c == '"')
            {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20)
            {
                return Fail("control character in string");
            }
            if (c != '\\')
            {
                out += c;
                continue;
            }
            if (mPos >= mText.size())
            {
                break;
            }
            const char e = mText[mPos++];
            switch (e)
            {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u':
            {
                std::optional<u32> cp = ParseHex4();
                if (!cp)
                {
                    return std::nullopt;
                }
                if (*cp >= 0xD800 && *cp <= 0xDBFF) // high surrogate: a low one must follow
                {
                    const std::optional<u32> low = Consume("\\u") ? ParseHex4() : std::nullopt;
                    if (!low || *low < 0xDC00 || *low > 0xDFFF)
                    {
                        return Fail("unpaired surrogate");
                    }
                    *cp = 0x10000 + ((*cp - 0xD800) << 10) + (*low - 0xDC00);
                }
                AppendUtf8(out, *cp);
                break;
            }
            default: return Fail("invalid escape");
            }
        }
        return Fail("unterminated string");
    }

    std::optional<Value> ParseNumber()
    {
        const size_t start = mPos;
        while (mPos < mText.size() &&
               (std::isdigit(static_cast<unsigned char>(mText[mPos])) || mText[mPos] == '.' || mText[mPos] == 'e' ||
                mText[mPos] == 'E' || mText[mPos] == '+' || mText[mPos] == '-'))
        {
            ++mPos;
        }
        double value = 0.0;
        const auto [end, ec] = std::from_chars(mText.data() + start, mText.data() + mPos, value);
        if (start == mPos || ec != std::errc() || end != mText.data() + mPos)
        {
            mPos = start;
            return Fail("invalid value");
        }
        return Value(value);
    }

    std::string_view mText;
    size_t mPos = 0;
    std::string mError;
};

} // namespace

std::optional<Value> Parse(std::string_view text, std::string* error)
{
    Parser parser(text);
    std::optional<Value> value = parser.Document();
    if (!value && error)
    {
        *error = parser.Error();
    }
    return value;
}

} // namespace air::json
