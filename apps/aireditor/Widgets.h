#pragma once

// Small ImGui helpers shared by the editor panels.

#include "air/core/Types.h"
#include "air/renderer/Parameters.h"

#include <imgui.h>

#include <string>
#include <string_view>
#include <vector>

namespace air::editor {

// Fixed-size history of a value for rolling graphs.
class History {
public:
    explicit History(size_t capacity = 300) : mValues(capacity, 0.0f) {}
    void Push(float value);
    const float* Data() const { return mValues.data(); }
    int Size() const { return int(mValues.size()); }
    int Offset() const { return int(mNext); } // index of the oldest value
    float Last() const { return mValues[(mNext + mValues.size() - 1) % mValues.size()]; }
    float Max() const;
    float Average() const;

private:
    std::vector<float> mValues;
    size_t mNext = 0;
    size_t mCount = 0;
};

// Rolling line graph with the current / average / max values in the label.
void PlotHistory(const char* label, const History& history, const char* unit, float height);

// One widget per parameter: checkbox, slider, or a combo when the description enumerates values
// ("0 lit, 1 normals, ..."). Returns true if anything changed (the set is Touch()ed).
bool EditParameters(renderer::ParameterSet& parameters);

// Parses "0 lit, 1 normals, 2 base color" into {"lit", "normals", "base color"}; empty if the
// text is not such a list.
std::vector<std::string> ParseEnumDescription(std::string_view description);

// A stable, readable color per name for timelines.
ImU32 ColorForName(std::string_view name);

// Horizontal bar with text centred on it, e.g. heap usage.
void UsageBar(float fraction, const char* overlay, ImU32 color = 0);

std::string FormatMs(double ms);

} // namespace air::editor
