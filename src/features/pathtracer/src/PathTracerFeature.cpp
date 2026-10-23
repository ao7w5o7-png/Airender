// Registers the path tracer as a RenderFeature named "pathtracer".

#include "air/features/pathtracer/PathTracer.h"
#include "air/renderer/RenderFeature.h"

#include "PathTracerShared.h"
#include "ReSTIRShared.h"

#include <algorithm>
#include <cmath>

namespace air::pt
{
namespace
{

class PathTracerFeature final : public renderer::RenderFeature
{
public:
    PathTracerFeature()
    {
        mParameters.Add("spp", &mSettings.samplesPerPixelPerFrame, "Samples per pixel traced each frame", 1, 4096);
        mParameters.Add("maxBounces", &mSettings.maxBounces, "Maximum path length (0 = emitters only)", 0, 64);
        mParameters.Add("targetSamples", &mSettings.targetSamples,
                        "Samples per pixel at which the image is final and tracing stops", 1, 1u << 20);
        mParameters.Add("accumulate", &mSettings.accumulate,
                        "Average samples across frames (off: per-frame noisy input for a denoiser)");
        mParameters.Add("guides", &mSettings.guides,
                        "Write denoiser guide textures (albedo, normals, depth, motion) when a feature reads them");
        mParameters.Add("nee", &mSettings.nextEventEstimation,
                        "Next event estimation with MIS (off: BSDF sampling only, for comparison)");
        mParameters.Add("lightTreeBounces", &mSettings.lightTreeBounces,
                        "Path vertices whose NEE chooses lights with the light tree (power, distance, orientation); "
                        "deeper ones choose by power. 0: by power only",
                        0, 64);
        mParameters.Add("cloudEvents", &mSettings.cloudEvents,
                        "Scattering events a path may take in the clouds before its light is given up", 0, 1u << 16);
        mParameters.Add("restirDi", &mSettings.restir.di,
                        "ReSTIR DI: resample the direct light at the first surface across pixels and frames "
                        "(0 off, 1 when not accumulating, 2 always)",
                        0, kReSTIRAlways);
        mParameters.Add("restirGi", &mSettings.restir.gi,
                        "ReSTIR GI: resample the indirect light at the first surface across pixels and frames "
                        "(0 off, 1 when not accumulating, 2 always)",
                        0, kReSTIRAlways);
        mParameters.Add("restirLights", &mSettings.restir.lightCandidates,
                        "ReSTIR initial candidates: light points chosen by power", 0, 64);
        mParameters.Add("restirBsdf", &mSettings.restir.bsdfCandidates,
                        "ReSTIR initial candidates: BSDF rays that may reach an emitter", 0, 8);
        mParameters.Add("restirSpatial", &mSettings.restir.spatialSamples, "ReSTIR neighbours reused per pixel", 0,
                        gpu::kReSTIRMaxSpatialSamples);
        mParameters.Add("restirDisocclusion", &mSettings.restir.disocclusionSamples,
                        "ReSTIR neighbours reused where a pixel has no temporal history", 0,
                        gpu::kReSTIRMaxSpatialSamples);
        mParameters.Add("restirRadius", &mSettings.restir.spatialRadius, "ReSTIR neighbour radius in pixels", 1.0f,
                        256.0f);
        mParameters.Add("restirHistory", &mSettings.restir.maxConfidence,
                        "ReSTIR temporal history cap, in frames' worth of candidates", 1, 254);
        mParameters.Add("restirUnbiased", &mSettings.restir.unbiased,
                        "ReSTIR MIS weights trace the visibility they need (off: fewer rays, darker penumbrae)");
        mParameters.Add("furnace", &mSettings.furnace,
                        "White furnace: white materials, lights off; energy conservation shows as a flat grey image");
        mParameters.Add("debugView", &mSettings.debugView,
                        "0 lit, 1 normals, 2 base color, 3 roughness, 4 metallic, 5 depth, 6 intersection heatmap, "
                        "7 bounce heatmap, 8 direct light, 9 indirect light",
                        0, gpu::kDebugViewCount - 1);
        mParameters.Add(
            "ser", &mSettings.shaderExecutionReordering,
            "Shader execution reordering (DXR 1.2) where the GPU reorders; slower with today's cheap shading");
        mParameters.Add("debugScale", &mSettings.debugScale,
                        "Full scale of the depth (metres) and intersection heatmap views", 1e-3f, 1e6f);
    }

    bool Initialize(const renderer::FeatureInitContext& context) override
    {
        // The Scene stage renders at the render resolution; an upscaler reconstructs the output.
        mSettings.width = context.renderWidth;
        mSettings.height = context.renderHeight;
        // Jitter phases for temporal upscalers: 8 x (output / render)^2 (NVIDIA DLSS guide), and
        // never fewer than the 32 used at native resolution.
        const float ratio = float(context.width) / float(std::max(context.renderWidth, 1u));
        mSettings.jitterPhases = std::max(32u, u32(std::ceil(8.0f * ratio * ratio)));
        mTracer = std::make_unique<PathTracer>(context.device, mSettings);
        return mTracer->Initialize(context.shaderDir);
    }

    void AddPasses(renderer::FrameContext& context) override
    {
        // Parameters write into mSettings; the tracer restarts accumulation when they change the image.
        PathTracerSettings& live = mTracer->MutableSettings();
        const u32 width = live.width;
        const u32 height = live.height;
        const u32 phases = live.jitterPhases;
        live = mSettings;
        live.width = width;
        live.height = height;
        live.jitterPhases = phases;
        // The guides cost about a tenth of the trace: only write them for a consumer (dlss).
        live.guides = mSettings.guides && context.Consumes(renderer::slots::kDiffuseAlbedo);
        mTracer->AddPasses(context.graph, context.scene, context.camera, context.blackboard, context.sceneColor);
    }

    renderer::FeatureStage GetStage() const override { return renderer::FeatureStage::Scene; }

    json::Value GetStatus() const override
    {
        json::Value status = json::Value::MakeObject();
        status["accumulatedSamples"] = mTracer ? mTracer->GetAccumulatedSamples() : 0u;
        status["converged"] = mTracer && mTracer->IsConverged();
        status["ser"] = mTracer && mTracer->UsesShaderExecutionReordering();
        status["restirDi"] = mTracer && mTracer->UsesReSTIRDI();
        status["restirGi"] = mTracer && mTracer->UsesReSTIRGI();
        return status;
    }

private:
    PathTracerSettings mSettings;
    std::unique_ptr<PathTracer> mTracer;
};

} // namespace
} // namespace air::pt

namespace air::features
{

void Register_pathtracer(renderer::FeatureRegistry& registry)
{
    registry.Register({"pathtracer", "Progressive DXR path tracer: GGX multi-scatter BSDF, NEE + MIS, physical units",
                       [] { return std::make_unique<pt::PathTracerFeature>(); }});
}

} // namespace air::features
