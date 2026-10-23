#include "TestHelpers.h"

using namespace airtest;

TEST(Example_RunsWithTransientPattern)
{
    Headless h({"example"});
    auto result = h.Frame();
    EXPECT(result.validationErrors.empty());
    EXPECT(result.graphStats.passCount == 2);
    EXPECT(result.graphStats.transientResourceCount == 1);
    EXPECT(ContainsLine(result.commands, "BeginEvent 'Example.Generate'"));
    EXPECT(ContainsLine(result.commands, "tex 'Example.Pattern' UAVCompute -> SRVCompute"));
}

TEST(Example_ComposesAfterPathTracer)
{
    Headless h({"pathtracer", "example"});
    EXPECT(h.runner->SetParameter("example.opacity", "0.25"));
    auto result = h.Frame();
    EXPECT(result.validationErrors.empty());
    EXPECT(result.graphStats.passCount == 4);
}
