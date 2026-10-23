#include "TestHelpers.h"

using namespace airtest;

TEST(PathTracer_AccumulatesAcrossFrames)
{
    Headless h({"pathtracer"});
    EXPECT(h.runner->SetParameter("pathtracer.spp", "2"));
    EXPECT(h.runner->SetParameter("pathtracer.targetSamples", "6"));

    std::vector<std::string> commands;
    for (int frame = 0; frame < 3; ++frame)
    {
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

TEST(PathTracer_ParameterChangeRestartsAccumulation)
{
    Headless h({"pathtracer"});
    h.Frame();
    h.Frame();
    EXPECT(h.runner->SetParameter("pathtracer.maxBounces", "2"));
    h.Frame();
    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"accumulatedSamples\":4") != std::string::npos); // one frame at default spp=4
}

TEST(PathTracer_RejectsInvalidParameters)
{
    Headless h({"pathtracer"});
    std::string error;
    EXPECT(!h.runner->SetParameter("pathtracer.maxBounces", "1000", &error));
    EXPECT(error.find("must be in") != std::string::npos);
    EXPECT(!h.runner->SetParameter("pathtracer.nope", "1", &error));
    EXPECT(!h.runner->SetParameter("missing.spp", "1", &error));
}

TEST(PathTracer_CameraMoveRestartsAccumulation)
{
    Headless h({"pathtracer"});
    h.Frame();
    h.Frame();
    h.camera.yaw += 0.1f;
    h.Frame();
    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"accumulatedSamples\":4") != std::string::npos);
}

TEST(PathTracer_StopsTracingWhenConverged)
{
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

namespace
{

// Stands in for a denoiser: declares the guides as consumed and reads them.
class GuideReader final : public air::renderer::RenderFeature
{
public:
    bool Initialize(const air::renderer::FeatureInitContext&) override { return true; }
    void AddPasses(air::renderer::FrameContext&) override {}
    std::vector<std::string_view> ConsumedSlots() const override
    {
        namespace slots = air::renderer::slots;
        return {slots::kDiffuseAlbedo, slots::kSpecularAlbedo, slots::kNormalRoughness, slots::kViewDepth,
                slots::kMotionVectors};
    }
};

} // namespace

// Guides cost about a tenth of the trace, so they are written only when a feature consumes them.
TEST(PathTracer_WritesGuidesForDenoisers)
{
    Headless h({"pathtracer"});
    EXPECT(h.Frame().graphText.find("PathTracer.DiffuseAlbedo") == std::string::npos); // nobody reads them
    EXPECT(h.runner->AddFeature("denoiser", std::make_unique<GuideReader>()));
    const auto result = h.Frame();
    for (const char* name : {"PathTracer.DiffuseAlbedo", "PathTracer.SpecularAlbedo", "PathTracer.NormalRoughness",
                             "PathTracer.ViewDepth", "PathTracer.MotionVectors"})
    {
        EXPECT(result.graphText.find(name) != std::string::npos);
    }
    EXPECT(h.runner->SetParameter("pathtracer.guides", "false"));
    EXPECT(h.Frame().graphText.find("PathTracer.ViewDepth") == std::string::npos);
}

TEST(PathTracer_DebugViewsAndFurnaceRun)
{
    Headless h({"pathtracer", "tonemap"});
    for (int view = 0; view < 10; ++view)
    {
        EXPECT(h.runner->SetParameter("pathtracer.debugView", std::to_string(view)));
        EXPECT(h.Frame().validationErrors.empty());
    }
    EXPECT(!h.runner->SetParameter("pathtracer.debugView", "10"));
    EXPECT(h.runner->SetParameter("pathtracer.furnace", "true"));
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "false"));
    EXPECT(h.Frame().validationErrors.empty());
}

// ReSTIR DI adds its three passes after the trace; from the second frame on the temporal pass reads
// the previous frame's primary hits and reservoirs, which swap with this frame's every frame.
TEST(PathTracer_ReSTIRResamplesAcrossFrames)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.restir", "1"));
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (int frame = 0; frame < 4; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        for (const char* pass :
             {"PathTracer.ReSTIR.Initial", "PathTracer.ReSTIR.Temporal", "PathTracer.ReSTIR.Spatial"})
        {
            EXPECT(CountLines(result.commands, std::string("BeginEvent '") + pass + "'") == 1);
        }
        // This frame writes one set of buffers; from the second frame on it also reads the other.
        const bool history = frame > 0;
        const char* current = frame % 2 == 0 ? "PrimaryHits[0]" : "PrimaryHits[1]";
        const char* previous = frame % 2 == 0 ? "PrimaryHits[1]" : "PrimaryHits[0]";
        EXPECT(result.graphText.find(current) != std::string::npos);
        EXPECT((result.graphText.find(previous) != std::string::npos) == history);
    }
    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"restir\":true") != std::string::npos);
}

// ReSTIR only replaces NEE where there is one: not for BSDF-only paths, the furnace or the views
// that show no light.
TEST(PathTracer_ReSTIROnlyRunsForLitViews)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.restir", "1"));
    auto restirPasses = [&] { return CountLines(h.Frame().commands, "BeginEvent 'PathTracer.ReSTIR."); };
    EXPECT(restirPasses() == 3);
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "1")); // normals
    EXPECT(restirPasses() == 0);
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "8")); // direct light
    EXPECT(restirPasses() == 3);
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.nee", "0"));
    EXPECT(restirPasses() == 0);
    EXPECT(h.runner->SetParameter("pathtracer.nee", "1"));
    EXPECT(h.runner->SetParameter("pathtracer.furnace", "1"));
    EXPECT(restirPasses() == 0);
    EXPECT(h.Frame().validationErrors.empty());
}
