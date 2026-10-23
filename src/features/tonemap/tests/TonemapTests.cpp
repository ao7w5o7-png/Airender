#include "TestHelpers.h"

using namespace airtest;

TEST(Tonemap_RunsInPlaceAfterPathTracer) {
    Headless h({"pathtracer", "tonemap"});
    const auto result = h.Frame();
    EXPECT(result.validationErrors.empty());
    EXPECT(ContainsLine(result.commands, "BeginEvent 'Tonemap'"));
    EXPECT(result.graphText.find("Tonemap") != std::string::npos);
}

TEST(Tonemap_RejectsUnknownOperator) {
    Headless h({"tonemap"});
    std::string error;
    EXPECT(!h.runner->SetParameter("tonemap.operator", "7", &error));
    EXPECT(h.runner->SetParameter("tonemap.operator", "2"));
    EXPECT(h.Frame().validationErrors.empty());
}
