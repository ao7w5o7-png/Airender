#include "TestHelpers.h"

using namespace airtest;

TEST(PathTracer_AccumulatesAcrossFrames) {
    Headless h({"pathtracer"});
    EXPECT(h.runner->SetParameter("pathtracer.spp", "2"));
    EXPECT(h.runner->SetParameter("pathtracer.targetSamples", "6"));

    std::vector<std::string> commands;
    for (int frame = 0; frame < 3; ++frame) {
        auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        commands.insert(commands.end(), result.commands.begin(), result.commands.end());
    }
    EXPECT(CountLines(commands, "DispatchRays 64x32x1") == 3);
    EXPECT(CountLines(commands, "Dispatch 8x4x1") == 3);

    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"accumulatedSamples\":6") != std::string::npos);
    EXPECT(status.find("\"converged\":true") != std::string::npos);
}

TEST(PathTracer_ParameterChangeRestartsAccumulation) {
    Headless h({"pathtracer"});
    h.Frame();
    h.Frame();
    EXPECT(h.runner->SetParameter("pathtracer.maxBounces", "2"));
    h.Frame();
    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"accumulatedSamples\":4") != std::string::npos); // one frame at default spp=4
}

TEST(PathTracer_RejectsInvalidParameters) {
    Headless h({"pathtracer"});
    std::string error;
    EXPECT(!h.runner->SetParameter("pathtracer.maxBounces", "1000", &error));
    EXPECT(error.find("must be in") != std::string::npos);
    EXPECT(!h.runner->SetParameter("pathtracer.nope", "1", &error));
    EXPECT(!h.runner->SetParameter("missing.spp", "1", &error));
}

TEST(PathTracer_CameraMoveRestartsAccumulation) {
    Headless h({"pathtracer"});
    h.Frame();
    h.Frame();
    h.camera.yaw += 0.1f;
    h.Frame();
    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"accumulatedSamples\":4") != std::string::npos);
}

TEST(PathTracer_StopsTracingWhenConverged) {
    Headless h({"pathtracer"});
    EXPECT(h.runner->SetParameter("pathtracer.spp", "4"));
    EXPECT(h.runner->SetParameter("pathtracer.targetSamples", "6"));
    EXPECT(CountLines(h.Frame().commands, "DispatchRays") == 1);
    EXPECT(CountLines(h.Frame().commands, "DispatchRays") == 1); // the last 2 samples
    const auto converged = h.Frame();
    EXPECT(CountLines(converged.commands, "DispatchRays") == 0);
    EXPECT(CountLines(converged.commands, "BeginEvent 'PathTracer.Resolve'") == 1);
    EXPECT(converged.validationErrors.empty());
}

TEST(PathTracer_WritesGuidesForDenoisers) {
    Headless h({"pathtracer"});
    const auto result = h.Frame();
    for (const char* name : {"PathTracer.DiffuseAlbedo", "PathTracer.SpecularAlbedo", "PathTracer.NormalRoughness",
                             "PathTracer.ViewDepth", "PathTracer.MotionVectors"}) {
        EXPECT(result.graphText.find(name) != std::string::npos);
    }
    EXPECT(h.runner->SetParameter("pathtracer.guides", "false"));
    EXPECT(h.Frame().graphText.find("PathTracer.ViewDepth") == std::string::npos);
}

TEST(PathTracer_DebugViewsAndFurnaceRun) {
    Headless h({"pathtracer", "tonemap"});
    for (int view = 0; view < 10; ++view) {
        EXPECT(h.runner->SetParameter("pathtracer.debugView", std::to_string(view)));
        EXPECT(h.Frame().validationErrors.empty());
    }
    EXPECT(!h.runner->SetParameter("pathtracer.debugView", "10"));
    EXPECT(h.runner->SetParameter("pathtracer.furnace", "true"));
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "false"));
    EXPECT(h.Frame().validationErrors.empty());
}
