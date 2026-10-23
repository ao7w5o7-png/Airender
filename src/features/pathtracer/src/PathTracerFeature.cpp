// Registers the path tracer as a RenderFeature named "pathtracer".

#include "air/features/pathtracer/PathTracer.h"
#include "air/renderer/RenderFeature.h"

#include "PathTracerShared.h"

namespace air::pt {
namespace {

class PathTracerFeature final : public renderer::RenderFeature {
public:
    PathTracerFeature() {
        mParameters.Add("spp", &mSettings.samplesPerPixelPerFrame, "Samples per pixel traced each frame", 1, 4096);
        mParameters.Add("maxBounces", &mSettings.maxBounces, "Maximum path length (0 = emitters only)", 0, 64);
        mParameters.Add("targetSamples", &mSettings.targetSamples,
                        "Samples per pixel at which the image is final and tracing stops", 1, 1u << 20);
        mParameters.Add("accumulate", &mSettings.accumulate,
                        "Average samples across frames (off: per-frame noisy input for a denoiser)");
        mParameters.Add("guides", &mSettings.guides, "Write denoiser guide textures (albedo, normals, depth, motion)");
        mParameters.Add("nee", &mSettings.nextEventEstimation,
                        "Next event estimation with MIS (off: BSDF sampling only, for comparison)");
        mParameters.Add("furnace", &mSettings.furnace,
                        "White furnace: white materials, lights off; energy conservation shows as a flat grey image");
        mParameters.Add("debugView", &mSettings.debugView,
                        "0 lit, 1 normals, 2 base color, 3 roughness, 4 metallic, 5 depth, 6 intersection heatmap, "
                        "7 bounce heatmap, 8 direct light, 9 indirect light",
                        0, gpu::kDebugViewCount - 1);
        mParameters.Add("debugScale", &mSettings.debugScale,
                        "Full scale of the depth (metres) and intersection heatmap views", 1e-3f, 1e6f);
    }

    bool Initialize(const renderer::FeatureInitContext& context) override {
        mSettings.width = context.width;
        mSettings.height = context.height;
        mTracer = std::make_unique<PathTracer>(context.device, mSettings);
        return mTracer->Initialize(context.shaderDir);
    }

    void AddPasses(renderer::FrameContext& context) override {
        // Parameters write into mSettings; the tracer restarts accumulation when they change the image.
        PathTracerSettings& live = mTracer->MutableSettings();
        const u32 width = live.width;
        const u32 height = live.height;
        live = mSettings;
        live.width = width;
        live.height = height;
        mTracer->AddPasses(context.graph, context.scene, context.camera, context.blackboard, context.output);
    }

    json::Value GetStatus() const override {
        json::Value status = json::Value::MakeObject();
        status["accumulatedSamples"] = mTracer ? mTracer->GetAccumulatedSamples() : 0u;
        status["converged"] = mTracer && mTracer->IsConverged();
        return status;
    }

private:
    PathTracerSettings mSettings;
    std::unique_ptr<PathTracer> mTracer;
};

} // namespace
} // namespace air::pt

namespace air::features {

void Register_pathtracer(renderer::FeatureRegistry& registry) {
    registry.Register({"pathtracer", "Progressive DXR path tracer: GGX multi-scatter BSDF, NEE + MIS, physical units",
                       [] { return std::make_unique<pt::PathTracerFeature>(); }});
}

} // namespace air::features
