#include "TestHelpers.h"

using namespace airtest;

// Behind the path tracer in real time TAA writes the output (from its history, which it copies), on
// accumulated frames it stands aside and FrameRunner's own passes fill the output.
TEST(Taa_FiltersRealTimeFramesOnly)
{
    Headless h({"pathtracer", "taa", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (int frame = 0; frame < 3; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        EXPECT(CountLines(result.commands, "BeginEvent 'TAA'") == 1);
        EXPECT(CountLines(result.commands, "BeginEvent 'TAA.Output'") == 1);
        // From the second frame on it reads its last result.
        const std::string past = "read  TAA.History[" + std::to_string((frame + 1) % 2) + "] as SRVCompute";
        EXPECT((result.graphText.find(past) != std::string::npos) == (frame > 0));
    }
    EXPECT(h.runner->DescribeFeatures().Dump(0).find("\"filtered\":true") != std::string::npos);
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "1"));
    const auto accumulated = h.Frame();
    EXPECT(accumulated.validationErrors.empty());
    EXPECT(CountLines(accumulated.commands, "BeginEvent 'TAA'") == 0);
}
