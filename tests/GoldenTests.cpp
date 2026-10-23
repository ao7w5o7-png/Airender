// Golden snapshot tests: the full command stream and render graph of reference scenarios are
// checked into tests/golden/. Any change in pass order, barriers, culling, aliasing or dispatch
// sizes shows up as a readable text diff in review. Update with AIR_UPDATE_GOLDEN=1.

#include "TestHelpers.h"

using namespace airtest;

namespace
{

std::string Snapshot(Headless& h, int frames)
{
    std::string out;
    for (int frame = 0; frame < frames; ++frame)
    {
        auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        out += "=== frame " + std::to_string(frame) + " graph\n" + result.graphText;
        out += "=== frame " + std::to_string(frame) + " commands\n";
        for (const std::string& line : result.commands)
        {
            out += line + '\n';
        }
    }
    return out;
}

} // namespace

TEST(Golden_PathTracer)
{
    Headless h({"pathtracer"});
    EXPECT(CheckGolden("pathtracer.txt", Snapshot(h, 2)));
}

TEST(Golden_Example)
{
    Headless h({"example"});
    EXPECT(CheckGolden("example.txt", Snapshot(h, 1)));
}

TEST(Golden_PathTracerThenExample)
{
    Headless h({"pathtracer", "example"});
    EXPECT(CheckGolden("pathtracer_example.txt", Snapshot(h, 1)));
}
