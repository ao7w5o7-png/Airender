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
    for (int view = 0; view < 14; ++view)
    {
        EXPECT(h.runner->SetParameter("pathtracer.debugView", std::to_string(view)));
        EXPECT(h.Frame().validationErrors.empty());
    }
    EXPECT(!h.runner->SetParameter("pathtracer.debugView", "14"));
    EXPECT(h.runner->SetParameter("pathtracer.furnace", "true"));
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "false"));
    EXPECT(h.Frame().validationErrors.empty());
}

// ReSTIR DI adds its three passes after the trace; from the second frame on the temporal pass reads
// the previous frame's primary hits and reservoirs, which swap with this frame's every frame.
TEST(PathTracer_ReSTIRResamplesAcrossFrames)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.restirGi", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (int frame = 0; frame < 4; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        for (const char* pass :
             {"PathTracer.ReSTIRDI.Initial", "PathTracer.ReSTIRDI.Temporal", "PathTracer.ReSTIRDI.Spatial"})
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
    EXPECT(status.find("\"restirDi\":true") != std::string::npos);
    EXPECT(status.find("\"restirGi\":false") != std::string::npos);
}

// ReSTIR runs by default only in real time (no accumulation), and DI only replaces NEE where there
// is one: not for BSDF-only paths, the furnace or the views that show no light.
TEST(PathTracer_ReSTIROnlyRunsForLitViews)
{
    Headless h({"pathtracer", "tonemap"});
    auto restirPasses = [&] { return CountLines(h.Frame().commands, "BeginEvent 'PathTracer.ReSTIR"); };
    EXPECT(restirPasses() == 0); // accumulating
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    EXPECT(restirPasses() == 5); // DI and GI
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "1"));
    EXPECT(h.runner->SetParameter("pathtracer.restirGi", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.restirDi", "2")); // also when accumulating
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

// ReSTIR GI: the trace writes the initial reservoirs, two passes reuse them; it runs with or
// without ReSTIR DI, and also without lights (the furnace), where all light is indirect or sky.
TEST(PathTracer_ReSTIRGIResamplesTheIndirectLight)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.restirDi", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (int frame = 0; frame < 3; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        EXPECT(CountLines(result.commands, "BeginEvent 'PathTracer.ReSTIRGI.Temporal'") == 1);
        EXPECT(CountLines(result.commands, "BeginEvent 'PathTracer.ReSTIRGI.Spatial'") == 1);
        EXPECT(CountLines(result.commands, "BeginEvent 'PathTracer.ReSTIRDI.") == 0);
        EXPECT(result.graphText.find("PathTracer.ReSTIRGI.InitialReservoirs") != std::string::npos);
        // History from the second frame on.
        EXPECT((result.graphText.find(frame % 2 == 0 ? "ReSTIRGI.History[1]" : "ReSTIRGI.History[0]") !=
                std::string::npos) == (frame > 0));
    }
    EXPECT(h.runner->SetParameter("pathtracer.restirDi", "1"));
    const auto both = h.Frame();
    EXPECT(both.validationErrors.empty());
    EXPECT(CountLines(both.commands, "BeginEvent 'PathTracer.ReSTIR") == 5);
    EXPECT(h.runner->SetParameter("pathtracer.furnace", "1"));
    const auto furnace = h.Frame();
    EXPECT(furnace.validationErrors.empty());
    EXPECT(CountLines(furnace.commands, "BeginEvent 'PathTracer.ReSTIRGI.") == 2);
    EXPECT(CountLines(furnace.commands, "BeginEvent 'PathTracer.ReSTIRDI.") == 0);
    const auto status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"restirGi\":true") != std::string::npos);
}

// The neural radiance cache: its trace kernel, the training paths' render queries and the frame's
// optimizer steps (on the matrix units where the device has them: the Null device pretends), in
// place of ReSTIR GI; off again, the plain kernel runs.
TEST(PathTracer_RadianceCacheTrainsEveryFrame)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.nrc", "1"));
    for (int frame = 0; frame < 3; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        EXPECT(CountLines(result.commands, "BeginEvent 'NRC.Resolve'") == 1);
        EXPECT(CountLines(result.commands, "BeginEvent 'NRC.Train[") == 4);
        EXPECT(CountLines(result.commands, "BeginEvent 'NRC.Mlp.Optimize'") == 4);
        EXPECT(CountLines(result.commands, "BeginEvent 'NRC.Mlp.ConvertGradients'") == 4);
        // The uploaded weights' conversion comes first, on the first frame only.
        EXPECT(CountLines(result.commands, "BeginEvent 'NRC.Mlp.ConvertWeights'") == (frame == 0 ? 5u : 4u));
        EXPECT(CountLines(result.commands, "ConvertMatrix ") == (frame == 0 ? 27u : 24u));
        EXPECT(CountLines(result.commands, "BeginEvent 'PathTracer.ReSTIRGI.") == 0);
        EXPECT(CountLines(result.commands, "BeginEvent 'PathTracer.ReSTIRDI.") == 3);
    }
    std::string status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"nrc\":true") != std::string::npos);
    EXPECT(status.find("\"nrcLinearAlgebra\":true") != std::string::npos);
    EXPECT(status.find("\"nrcSteps\":12") != std::string::npos);

    // float32 kernels: no matrices to convert.
    EXPECT(h.runner->SetParameter("pathtracer.nrcLinearAlgebra", "0"));
    const auto portable = h.Frame();
    EXPECT(portable.validationErrors.empty());
    EXPECT(CountLines(portable.commands, "BeginEvent 'NRC.Train[") == 4);
    EXPECT(CountLines(portable.commands, "ConvertMatrix ") == 0);
    status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"nrcLinearAlgebra\":false") != std::string::npos);

    // The cache's view, and the direct light (no cache lookups to resolve there).
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "10"));
    EXPECT(CountLines(h.Frame().commands, "BeginEvent 'NRC.Resolve'") == 0);
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "8"));
    EXPECT(CountLines(h.Frame().commands, "BeginEvent 'NRC.Train[") == 4);
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "1")); // normals: no light, no cache
    EXPECT(CountLines(h.Frame().commands, "BeginEvent 'NRC.") == 0);

    EXPECT(h.runner->SetParameter("pathtracer.debugView", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.nrc", "0"));
    const auto off = h.Frame();
    EXPECT(off.validationErrors.empty());
    EXPECT(CountLines(off.commands, "BeginEvent 'NRC.") == 0);
    EXPECT(CountLines(off.commands, "BeginEvent 'PathTracer.ReSTIRGI.") == 2);
}

// A volume (smoke in the Cornell box): the plain kernel traces it, while ReSTIR and the radiance
// cache, which do not model media, step aside until it is gone.
TEST(PathTracer_VolumeTurnsReSTIRAndTheCacheOff)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    h.scene->SetVolumes({{.enabled = true, .position = {0.0f, 0.0f, 0.0f}, .size = 1.2f}});
    for (int frame = 0; frame < 2; ++frame)
    {
        const auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
        EXPECT(CountLines(result.commands, "BeginEvent 'PathTracer.ReSTIR") == 0);
        EXPECT(CountLines(result.commands, "DispatchRays 64x32x1") == 1);
    }
    EXPECT(h.scene->Get().volumes.size() == 1 && h.scene->Get().volumes[0].enabled == 1);
    EXPECT(h.runner->SetParameter("pathtracer.nrc", "1"));
    const auto cache = h.Frame();
    EXPECT(cache.validationErrors.empty());
    EXPECT(CountLines(cache.commands, "BeginEvent 'NRC.") == 0);
    std::string status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"restirDi\":false") != std::string::npos && status.find("\"nrc\":false") != std::string::npos);

    h.scene->SetVolumes({});
    const auto without = h.Frame();
    EXPECT(without.validationErrors.empty());
    EXPECT(CountLines(without.commands, "BeginEvent 'NRC.Train[") == 4);
    EXPECT(CountLines(without.commands, "BeginEvent 'PathTracer.ReSTIRDI.") == 3);
}

// Real-time volumes (pathtracer.volumeMarch, docs/architecture.md §8.13): not accumulating, the volume
// is ray marched over the trace's samples. The trace runs the kernel that lights the surfaces through
// the moment shadow maps, the maps render on the compute queue when the scene changes (not every
// frame), the probes run there beside the trace every frame (they wait for nothing on the graphics
// queue but the frame's start), and the march adds to the accumulation. Its debug views run.
// Accumulating, the volume is tracked again.
TEST(PathTracer_RealTimeVolumesAreMarched)
{
    Headless h({"pathtracer", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    h.scene->SetVolumes({{.enabled = true, .position = {0.0f, 0.0f, 0.0f}, .size = 1.2f}});
    const auto first = h.Frame();
    EXPECT(first.validationErrors.empty());
    EXPECT(CountLines(first.commands, "SetPipeline pipeline 'PathTracer.TraceVolumeShadows'") == 1);
    EXPECT(CountLines(first.commands, "[Compute] BeginEvent 'VolumeMarch.Shadows'") == 1);
    EXPECT(CountLines(first.commands, "[Compute] BeginEvent 'VolumeMarch.Probes'") == 1);
    EXPECT(CountLines(first.commands, "BeginEvent 'VolumeMarch.March'") == 1);
    const auto second = h.Frame();
    EXPECT(second.validationErrors.empty());
    EXPECT(CountLines(second.commands, "BeginEvent 'VolumeMarch.Shadows'") == 0);
    EXPECT(CountLines(second.commands, "[Compute] BeginEvent 'VolumeMarch.Probes'") == 1);
    EXPECT(second.graphText.find("Compute: [1], waits for (0)\n") != std::string::npos);
    const std::string status = h.runner->DescribeFeatures().Dump(0);
    EXPECT(status.find("\"volumeMarch\":true") != std::string::npos);
    EXPECT(status.find("\"volumeMarchLights\":1") != std::string::npos); // the ceiling panel
    for (const char* view : {"11", "12", "13", "8", "9", "1"})
    {
        EXPECT(h.runner->SetParameter("pathtracer.debugView", view));
        EXPECT(h.Frame().validationErrors.empty());
    }
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "0"));
    // Another volume: the maps render again.
    h.scene->SetVolumes({{.enabled = true, .position = {0.0f, 0.0f, 0.0f}, .size = 1.2f},
                         {.enabled = true, .position = {0.3f, 0.2f, 0.0f}, .size = 0.8f, .seed = 4}});
    EXPECT(CountLines(h.Frame().commands, "BeginEvent 'VolumeMarch.Shadows'") == 1);
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "1"));
    const auto tracked = h.Frame();
    EXPECT(tracked.validationErrors.empty());
    EXPECT(CountLines(tracked.commands, "BeginEvent 'VolumeMarch.") == 0);
    EXPECT(h.runner->SetParameter("pathtracer.volumeMarch", "2"));
    EXPECT(CountLines(h.Frame().commands, "BeginEvent 'VolumeMarch.March'") == 1);
}
