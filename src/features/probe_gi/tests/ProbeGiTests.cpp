#include "TestHelpers.h"

using namespace airtest;

namespace
{

double StatusNumber(const air::renderer::FrameRunner& runner, const char* key)
{
    const air::renderer::RenderFeature* feature = runner.FindFeature("probe_gi");
    if (!feature)
    {
        return -1.0;
    }
    const air::json::Value status = feature->GetStatus();
    const air::json::Value* value = status.Find(key);
    return value ? value->AsNumber(-1.0) : -1.0;
}

// Stands in for a denoiser of the surfaces' signals (nrd, which needs D3D12).
class SignalReader final : public air::renderer::RenderFeature
{
public:
    bool Initialize(const air::renderer::FeatureInitContext&) override { return true; }
    void AddPasses(air::renderer::FrameContext&) override {}
    std::vector<std::string_view> ConsumedSlots() const override
    {
        namespace slots = air::renderer::slots;
        return {slots::kDiffuseAlbedo, slots::kSpecularAlbedo, slots::kNormalRoughness, slots::kViewDepth,
                slots::kMotionVectors, slots::kDiffuseSignal,  slots::kSpecularSignal,  slots::kSurfaceTransmittance};
    }
};

// The barriers and accesses of one pass in RenderGraph::DumpText(): the lines indented under its own.
std::string PassAccesses(const std::string& graphText, const std::string& pass)
{
    size_t end = graphText.find("] " + pass + " (");
    if (end == std::string::npos)
    {
        return {};
    }
    const size_t begin = end;
    while ((end = graphText.find('\n', end)) != std::string::npos && graphText.compare(end + 1, 6, "      ") == 0)
    {
        ++end;
    }
    return graphText.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
}

} // namespace

// The path tracer leaves the indirect light to probe_gi, which builds its voxels, lights them, updates its
// cache, traces its probes and adds the result: all without validation errors.
TEST(ProbeGi_AddsTheIndirectLight)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    EXPECT(h.runner->SetParameter("pathtracer.spp", "1"));
    auto first = h.Frame();
    EXPECT(first.validationErrors.empty());
    // The first frame builds every cascade.
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Voxelize[0].Distance'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Voxelize[5].BricksZ'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Probes.Trace'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Probes.Adapt'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Probes.RayFilter'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Reflections.Trace'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Reflections.Resolve'"));
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Composite'"));
    // The harmonics' own spatial filter is off by default (the rays' filter replaces it).
    EXPECT(!ContainsLine(first.commands, "BeginEvent 'ProbeGi.Probes.Filter'"));
    // The path tracer stops at the first surface's BSDF sample: no ReSTIR GI.
    EXPECT(!ContainsLine(first.commands, "ReSTIRGI"));
    auto second = h.Frame();
    EXPECT(second.validationErrors.empty());
    // Nothing moved: no cascade is built again; the voxels of every cascade are lit and the cache updated in
    // one dispatch each, and the counters copied for the pools' growth.
    EXPECT(!ContainsLine(second.commands, "ProbeGi.Voxelize["));
    EXPECT(ContainsLine(second.commands, "BeginEvent 'ProbeGi.VoxelLight'"));
    EXPECT(ContainsLine(second.commands, "BeginEvent 'ProbeGi.Cache.Trace'"));
    EXPECT(ContainsLine(second.commands, "BeginEvent 'ProbeGi.Counters'"));
    EXPECT(ContainsLine(second.commands, "BeginEvent 'ProbeGi.Temporal'"));
    // No cascade moved lately: no voxels are new.
    EXPECT(!ContainsLine(second.commands, "BeginEvent 'ProbeGi.VoxelLight.Fresh'"));
}

// The screen probes keep their harmonics over frames: the first frame starts the histories, the next ones
// read last frame's probes and harmonics; a new probe tile (another count of probes) starts over.
TEST(ProbeGi_ProbesAccumulateOverFrames)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    auto first = h.Frame();
    EXPECT(first.validationErrors.empty());
    EXPECT(ContainsLine(first.commands, "BeginEvent 'ProbeGi.Probes.Temporal'"));
    EXPECT(first.graphText.find("ProbeGi.ProbeShHistory[1]") == std::string::npos);
    auto second = h.Frame();
    EXPECT(second.validationErrors.empty());
    EXPECT(second.graphText.find("ProbeGi.ProbeShHistory[0]") != std::string::npos);
    EXPECT(second.graphText.find("ProbeGi.ProbeShHistory[1]") != std::string::npos);
    EXPECT(second.graphText.find("ProbeGi.Probes[1]") != std::string::npos);
    EXPECT(h.runner->SetParameter("probe_gi.probeTile", "8"));
    auto retiled = h.Frame();
    EXPECT(retiled.validationErrors.empty());
    EXPECT(retiled.graphText.find("ProbeGi.ProbeShHistory[1]") == std::string::npos);
    EXPECT(h.runner->SetParameter("probe_gi.probeBlend", "1"));
    EXPECT(h.Frame().validationErrors.empty());
}

// A camera that moves past a cascade's slack moves that cascade: it voxelizes only the bricks it did not
// cover before (scroll), or all of them again (scroll off).
TEST(ProbeGi_CascadesFollowTheCamera)
{
    for (const char* scroll : {"1", "0"})
    {
        Headless h({"pathtracer", "probe_gi"});
        EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
        EXPECT(h.runner->SetParameter("probe_gi.scroll", scroll));
        h.Frame();
        EXPECT(StatusNumber(*h.runner, "fullRebuilds") == 6.0);
        h.camera.position.x += 2.0f; // 40 voxels of the finest cascade
        auto moved = h.Frame();
        EXPECT(moved.validationErrors.empty());
        EXPECT(ContainsLine(moved.commands, "BeginEvent 'ProbeGi.Voxelize[0].Clear'"));
        EXPECT(ContainsLine(moved.commands, "BeginEvent 'ProbeGi.Voxelize[0].Resolve'"));
        EXPECT(ContainsLine(moved.commands, "BeginEvent 'ProbeGi.Cache[0].Reset'"));
        EXPECT(StatusNumber(*h.runner, "rebuilds") == 7.0);
        EXPECT(StatusNumber(*h.runner, "fullRebuilds") == (scroll[0] == '1' ? 6.0 : 7.0));
        // The voxels new where it scrolled are lit every frame while they are young; a whole rebuild lights all.
        EXPECT(ContainsLine(moved.commands, "BeginEvent 'ProbeGi.VoxelLight.Fresh'") == (scroll[0] == '1'));
        // The coarser cascades the move took past their slack follow, one a frame, and then nothing moves.
        auto still = h.Frame();
        for (int i = 0; i < 6 && ContainsLine(still.commands, "ProbeGi.Voxelize["); ++i)
        {
            EXPECT(still.validationErrors.empty());
            still = h.Frame();
        }
        EXPECT(still.validationErrors.empty());
        EXPECT(!ContainsLine(still.commands, "ProbeGi.Voxelize["));
        EXPECT(scroll[0] == '1' ? StatusNumber(*h.runner, "fullRebuilds") == 6.0
                                : StatusNumber(*h.runner, "fullRebuilds") >= 7.0);
    }
}

// A jump further than a cascade is wide leaves nothing to keep: everything is voxelized again.
TEST(ProbeGi_FarJumpRebuildsWhole)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    EXPECT(h.runner->SetParameter("probe_gi.cascades", "1"));
    h.Frame();
    h.camera.position.z += 20.0f; // 400 voxels: past the cascade's 128
    auto moved = h.Frame();
    EXPECT(moved.validationErrors.empty());
    EXPECT(StatusNumber(*h.runner, "rebuilds") == 2.0);
    EXPECT(StatusNumber(*h.runner, "fullRebuilds") == 2.0);
}

// The views of the voxel scene replace the frame and mark it as display-referred where they show values.
TEST(ProbeGi_ViewsOfTheVoxels)
{
    Headless h({"pathtracer", "probe_gi", "tonemap"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    for (const char* view : {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
    {
        EXPECT(h.runner->SetParameter("probe_gi.view", view));
        auto result = h.Frame();
        EXPECT(result.validationErrors.empty());
    }
    EXPECT(h.runner->SetParameter("probe_gi.view", "4"));
    auto voxels = h.Frame();
    EXPECT(ContainsLine(voxels.commands, "BeginEvent 'ProbeGi.VoxelView'"));
    EXPECT(!ContainsLine(voxels.commands, "BeginEvent 'ProbeGi.Composite'"));
}

// The radiance cache as the far field of the probes' and the voxels' rays, and the single bounce check.
TEST(ProbeGi_CacheAndOneBounce)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    EXPECT(h.runner->SetParameter("probe_gi.cacheDistance", "8"));
    EXPECT(h.runner->SetParameter("probe_gi.radiosityDistance", "16"));
    auto cached = h.Frame();
    EXPECT(cached.validationErrors.empty());
    EXPECT(h.runner->SetParameter("probe_gi.multiBounce", "0"));
    EXPECT(h.runner->SetParameter("probe_gi.radiosityRays", "0"));
    auto single = h.Frame();
    EXPECT(single.validationErrors.empty());
}

// Without probe_gi the path tracer keeps its indirect light; its reference view of what probe_gi adds runs.
TEST(ProbeGi_PathTracerReferenceView)
{
    Headless h({"pathtracer"});
    EXPECT(h.runner->SetParameter("pathtracer.debugView", "15"));
    auto result = h.Frame();
    EXPECT(result.validationErrors.empty());
}

// The probes' rays filtered per direction, the harmonics' filter, the traced reflections, the adaptive probes: each
// can be switched,
// without validation errors.
TEST(ProbeGi_FiltersAndReflections)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    EXPECT(h.runner->SetParameter("probe_gi.filterRadius", "1"));
    EXPECT(h.runner->SetParameter("probe_gi.rayFilterRadius", "0"));
    EXPECT(h.runner->SetParameter("probe_gi.reflections", "0"));
    EXPECT(h.runner->SetParameter("probe_gi.adaptiveProbes", "0"));
    auto result = h.Frame();
    EXPECT(result.validationErrors.empty());
    EXPECT(ContainsLine(result.commands, "BeginEvent 'ProbeGi.Probes.Filter'"));
    EXPECT(!ContainsLine(result.commands, "ProbeGi.Probes.Adapt"));
    EXPECT(!ContainsLine(result.commands, "ProbeGi.Reflections"));
    EXPECT(h.runner->SetParameter("probe_gi.reflections", "1"));
    auto traced = h.Frame();
    EXPECT(traced.validationErrors.empty());
    EXPECT(ContainsLine(traced.commands, "BeginEvent 'ProbeGi.Reflections.Trace'"));
}

// Under a denoiser of the surfaces' signals the GI joins them (else the denoiser leaves it noisy); without one
// the path tracer writes no signals, and a view of the GI keeps out of them.
TEST(ProbeGi_AddsToTheDenoisersSignals)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    auto alone = h.Frame();
    EXPECT(alone.validationErrors.empty());
    EXPECT(alone.graphText.find("PathTracer.DiffuseSignal") == std::string::npos);
    EXPECT(h.runner->AddFeature("denoiser", std::make_unique<SignalReader>()));
    auto denoised = h.Frame();
    EXPECT(denoised.validationErrors.empty());
    const std::string composite = PassAccesses(denoised.graphText, "ProbeGi.Composite");
    EXPECT(composite.find("write PathTracer.DiffuseSignal") != std::string::npos);
    EXPECT(composite.find("write PathTracer.SpecularSignal") != std::string::npos);
    EXPECT(h.runner->SetParameter("probe_gi.view", "1"));
    auto view = h.Frame();
    EXPECT(view.validationErrors.empty());
    EXPECT(PassAccesses(view.graphText, "ProbeGi.Composite").find("Signal") == std::string::npos);
}

// A change of the lights (here the sky's radiance) lights every voxel and cache probe again at once.
TEST(ProbeGi_RelightsWhenTheLightsChange)
{
    Headless h({"pathtracer", "probe_gi"});
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    h.Frame();
    h.Frame();
    EXPECT(StatusNumber(*h.runner, "relights") == 0.0);
    h.scene->SetEnvironment(air::renderer::float3(0.5f, 0.6f, 0.8f));
    auto relit = h.Frame();
    EXPECT(relit.validationErrors.empty());
    EXPECT(StatusNumber(*h.runner, "relights") == 1.0);
}