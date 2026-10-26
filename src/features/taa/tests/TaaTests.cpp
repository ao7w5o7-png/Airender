#include "TestHelpers.h"

using namespace airtest;

// Behind the path tracer in real time TAA writes the output (from its history, which it copies), on
// accumulated frames it stands aside and FrameRunner's own passes fill the output. It first updates
// its fast history (TAA.Fast), whose neighbourhood the result's history is clipped to.
TEST(Taa_FiltersRealTimeFramesOnly)
{
    Headless h({"pathtracer", "taa", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (int frame = 0; frame < 3; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        EXPECT(CountLines(result.commands, "BeginEvent 'TAA.Fast'") == 1);
        EXPECT(CountLines(result.commands, "BeginEvent 'TAA'") == 1);
        EXPECT(CountLines(result.commands, "BeginEvent 'TAA.Output'") == 1);
        // From the second frame on it reads its last result and fast history.
        const std::string slot = std::to_string((frame + 1) % 2);
        EXPECT((result.graphText.find("read  TAA.History[" + slot + "] as SRVCompute") != std::string::npos) ==
               (frame > 0));
        EXPECT((result.graphText.find("read  TAA.FastHistory[" + slot + "] as SRVCompute") != std::string::npos) ==
               (frame > 0));
        // The result reads this frame's fast history.
        const size_t taa = result.graphText.find("TAA (Compute)");
        EXPECT(taa != std::string::npos &&
               result.graphText.find("read  TAA.FastHistory[" + std::to_string(frame % 2) + "] as SRVCompute", taa) !=
                   std::string::npos);
    }
    EXPECT(h.runner->DescribeFeatures().Dump(0).find("\"filtered\":true") != std::string::npos);
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "1"));
    const auto accumulated = h.Frame();
    EXPECT(accumulated.validationErrors.empty());
    EXPECT(CountLines(accumulated.commands, "BeginEvent 'TAA'") == 0);
}

// A still camera lets the history keep the mean of noisy frames (a wide box around the fast history);
// a moving one clips it to the frame's own samples, also where the motion vectors do not move (the sky
// behind smoke).
TEST(Taa_KnowsWhenTheCameraMoves)
{
    Headless h({"pathtracer", "taa", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    auto moved = [&] { return h.runner->DescribeFeatures().Dump(0).find("\"cameraMoved\":true") != std::string::npos; };
    EXPECT(h.Frame().validationErrors.empty());
    EXPECT(!moved()); // no history yet
    EXPECT(h.Frame().validationErrors.empty());
    EXPECT(!moved());
    h.camera.position.x += 0.01f;
    EXPECT(h.Frame().validationErrors.empty());
    EXPECT(moved());
    EXPECT(h.Frame().validationErrors.empty());
    EXPECT(!moved());
}
