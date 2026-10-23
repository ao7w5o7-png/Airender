// Registers the path tracer as a RenderFeature named "pathtracer".

#include "air/features/pathtracer/PathTracer.h"
#include "air/renderer/RenderFeature.h"

namespace air::pt {
namespace {

class PathTracerFeature final : public renderer::RenderFeature {
public:
    PathTracerFeature() {
        mParameters.Add("spp", &mSettings.samplesPerPixelPerFrame, "Samples per pixel traced each frame", 1, 4096);
        mParameters.Add("maxBounces", &mSettings.maxBounces, "Maximum path length", 0, 64);
        mParameters.Add("targetSamples", &mSettings.targetSamples, "Samples per pixel at which the image is final", 1,
                        1u << 20);
        mParameters.Add("exposure", &mSettings.exposure, "Linear exposure applied before tonemapping", 0.0f, 1000.0f);
    }

    bool Initialize(const renderer::FeatureInitContext& context) override {
        mSettings.width = context.width;
        mSettings.height = context.height;
        mTracer = std::make_unique<PathTracer>(context.device, mSettings);
        return mTracer->Initialize(context.shaderDir);
    }

    void AddPasses(renderer::FrameContext& context) override {
        // Parameters write into mSettings; push them to the tracer and restart accumulation when
        // anything that changes the converged image changed.
        PathTracerSettings& live = mTracer->MutableSettings();
        const bool imageChanged = live.maxBounces != mSettings.maxBounces || context.scene.version != mSceneVersion;
        live.samplesPerPixelPerFrame = mSettings.samplesPerPixelPerFrame;
        live.maxBounces = mSettings.maxBounces;
        live.targetSamples = mSettings.targetSamples;
        live.exposure = mSettings.exposure;
        if (imageChanged) {
            mTracer->ResetAccumulation();
            mSceneVersion = context.scene.version;
        }
        mTracer->AddPasses(context.graph, context.scene, context.output);
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
    u64 mSceneVersion = ~0ull;
};

} // namespace
} // namespace air::pt

namespace air::features {

void Register_pathtracer(renderer::FeatureRegistry& registry) {
    registry.Register({"pathtracer", "Progressive hardware ray traced path tracer (offline / film quality)",
                       [] { return std::make_unique<pt::PathTracerFeature>(); }});
}

} // namespace air::features
