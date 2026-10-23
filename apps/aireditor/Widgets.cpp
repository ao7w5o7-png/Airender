#include "Widgets.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <format>
#include <numeric>

namespace air::editor
{

void History::Push(float value)
{
    mValues[mNext] = value;
    mNext = (mNext + 1) % mValues.size();
    mCount = std::min(mCount + 1, mValues.size());
}

float History::Max() const
{
    return mCount == 0 ? 0.0f : *std::max_element(mValues.begin(), mValues.end());
}

float History::Average() const
{
    if (mCount == 0)
    {
        return 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < mCount; ++i)
    {
        sum += mValues[(mNext + mValues.size() - 1 - i) % mValues.size()];
    }
    return sum / float(mCount);
}

void PlotHistory(const char* label, const History& history, const char* unit, float height)
{
    const std::string overlay = std::format("{} {:.2f} {} (avg {:.2f}, max {:.2f})", label, history.Last(), unit,
                                            history.Average(), history.Max());
    ImGui::PushID(label);
    ImGui::PlotLines("##plot", history.Data(), history.Size(), history.Offset(), overlay.c_str(), 0.0f,
                     std::max(history.Max() * 1.15f, 1e-3f), ImVec2(-1.0f, height));
    ImGui::PopID();
}

std::vector<std::string> ParseEnumDescription(std::string_view text)
{
    std::vector<std::string> names;
    size_t pos = 0;
    while (pos < text.size())
    {
        size_t end = text.find(',', pos);
        if (end == std::string_view::npos)
        {
            end = text.size();
        }
        std::string_view item = text.substr(pos, end - pos);
        while (!item.empty() && item.front() == ' ')
        {
            item.remove_prefix(1);
        }
        size_t digits = 0;
        while (digits < item.size() && std::isdigit(static_cast<unsigned char>(item[digits])))
        {
            ++digits;
        }
        if (digits == 0 || digits >= item.size() || item[digits] != ' ' ||
            std::stoul(std::string(item.substr(0, digits))) != names.size())
        {
            return {};
        }
        names.emplace_back(item.substr(digits + 1));
        pos = end + 1;
    }
    return names.size() >= 2 ? names : std::vector<std::string>{};
}

bool EditParameters(renderer::ParameterSet& parameters)
{
    bool changed = false;
    for (const renderer::ParameterSet::Entry& entry : parameters.Entries())
    {
        ImGui::PushID(entry.name.c_str());
        bool edited = false;
        if (auto* b = std::get_if<bool*>(&entry.value))
        {
            edited = ImGui::Checkbox(entry.name.c_str(), *b);
        }
        else if (auto* u = std::get_if<u32*>(&entry.value))
        {
            const std::vector<std::string> names = ParseEnumDescription(entry.description);
            if (!names.empty())
            {
                const u32 current = std::min<u32>(**u, u32(names.size() - 1));
                if (ImGui::BeginCombo(entry.name.c_str(), names[current].c_str()))
                {
                    for (u32 i = 0; i < names.size(); ++i)
                    {
                        if (ImGui::Selectable(names[i].c_str(), i == current))
                        {
                            **u = i;
                            edited = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            }
            else
            {
                int value = int(std::min<u32>(**u, 0x7FFFFFFF));
                const int lo = int(entry.min);
                const int hi = int(std::min(entry.max, double(0x7FFFFFFF)));
                if (hi - lo <= 64)
                {
                    edited = ImGui::SliderInt(entry.name.c_str(), &value, lo, hi);
                }
                else
                {
                    edited = ImGui::DragInt(entry.name.c_str(), &value, 1.0f, lo, hi, "%d",
                                            ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                }
                if (edited)
                {
                    **u = u32(std::clamp(value, lo, hi));
                }
            }
        }
        else if (auto* f = std::get_if<float*>(&entry.value))
        {
            const float lo = float(entry.min);
            const float hi = float(entry.max);
            const bool wide = hi - lo > 100.0f;
            edited =
                ImGui::DragFloat(entry.name.c_str(), *f, wide ? 0.05f : (hi - lo) / 500.0f, lo, hi, "%.3f",
                                 (wide && lo > 0.0f ? ImGuiSliderFlags_Logarithmic : 0) | ImGuiSliderFlags_AlwaysClamp);
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && !entry.description.empty())
        {
            ImGui::SetTooltip("%s", entry.description.c_str());
        }
        changed |= edited;
        ImGui::PopID();
    }
    if (changed)
    {
        parameters.Touch();
    }
    return changed;
}

ImU32 ColorForName(std::string_view name)
{
    u32 hash = 2166136261u;
    for (char c : name)
    {
        hash = (hash ^ u8(c)) * 16777619u;
    }
    const float hue = float(hash % 360) / 360.0f;
    float r = 0, g = 0, b = 0;
    ImGui::ColorConvertHSVtoRGB(hue, 0.55f, 0.85f, r, g, b);
    return ImGui::GetColorU32(ImVec4(r, g, b, 1.0f));
}

void UsageBar(float fraction, const char* overlay, ImU32 color)
{
    if (color != 0)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
    }
    ImGui::ProgressBar(std::clamp(fraction, 0.0f, 1.0f), ImVec2(-1.0f, 0.0f), overlay);
    if (color != 0)
    {
        ImGui::PopStyleColor();
    }
}

std::string FormatMs(double ms)
{
    return ms >= 1.0 ? std::format("{:.2f} ms", ms) : std::format("{:.1f} us", ms * 1000.0);
}

} // namespace air::editor
