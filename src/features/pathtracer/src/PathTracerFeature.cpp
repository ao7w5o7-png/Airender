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
        mParameters.Add("volumeEvents", &mSettings.volumeEvents,
                        "Scattering events a path may take in the scene's volume (besides its bounces) before its "
                        "light is given up",
                        0, 1u << 16);
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
        mParameters.Add("nrc", &mSettings.nrc.mode,
                        "Neural radiance cache: paths end where their footprint has spread and take the reflected "
                        "light from a network trained every frame; replaces ReSTIR GI (0 off, 1 when not "
                        "accumulating, 2 always)",
                        0, kNrcAlways);
        mParameters.Add("nrcSpread", &mSettings.nrc.spread,
                        "Radiance cache: a path ends where its spread exceeds this times the primary vertex's", 1e-5f,
                        100.0f);
        mParameters.Add("nrcTrainingPaths", &mSettings.nrc.trainingPaths,
                        "Radiance cache: training paths per frame (one per tile of pixels)", 64, 1u << 20);
        mParameters.Add("nrcSteps", &mSettings.nrc.steps, "Radiance cache: optimizer steps per frame", 1, 16);
        mParameters.Add("nrcLearningRate", &mSettings.nrc.learningRate, "Radiance cache: Adam's learning rate", 1e-5f,
                        1.0f);
        mParameters.Add("nrcUnbiased", &mSettings.nrc.unbiasedPeriod,
                        "Radiance cache: 1 in this many training paths is never ended by the cache", 1, 1024);
        mParameters.Add("nrcTableLog2", &mSettings.nrc.tableLog2,
                        "Radiance cache: hash grid entries per level, log2 (a change starts a new network)", 10, 22);
        mParameters.Add("nrcLinearAlgebra", &mSettings.nrc.linearAlgebra,
                        "Radiance cache on the GPU's matrix units (Shader Model 6.10 linear algebra) where available; "
                        "off: float32 kernels");
        VolumeMarchSettings& march = mSettings.volumeMarch;
        mParameters.Add("volumeMarch", &march.mode,
                        "Real-time volumes: ray march the volumes over the surfaces, lit through moment shadow maps, "
                        "instead of tracking them along every path (0 off, 1 when not accumulating, 2 always)",
                        0, kVolumeMarchAlways);
        mParameters.Add("volumeStep", &march.step, "Real-time volumes: the march's step, in voxels", 0.05f, 16.0f);
        mParameters.Add("volumeLevel", &march.level,
                        "Real-time volumes: the level of detail the march reads (0 the voxels, 1 and 2 averaged over "
                        "2^3 and 4^3)",
                        0, 2);
        mParameters.Add("volumeShadowSize", &march.shadowSize,
                        "Real-time volumes: texels per side of every moment shadow map", 16, 2048);
        mParameters.Add("volumeShadowLevel", &march.shadowLevel,
                        "Real-time volumes: the level of detail the shadow maps' marches read", 0, 2);
        mParameters.Add("volumeShadowStep", &march.shadowStep,
                        "Real-time volumes: the shadow maps' step, in voxels of their level", 0.05f, 16.0f);
        mParameters.Add("volumeLights", &march.lights,
                        "Real-time volumes: light groups (emissive instances) that light the volumes besides the sun, "
                        "those that matter most to them",
                        0, kMaxVolumeLights);
        mParameters.Add("volumeLightOrigins", &march.lightOrigins,
                        "Real-time volumes: shadow maps of a large emitter close to them, from points spread over it "
                        "(1: its centre only)",
                        1, kMaxLightOrigins);
        mParameters.Add("volumeAdaptive", &march.adaptive,
                        "Real-time volumes: adaptive sampling groups, waves deep in smoke or close to the camera share "
                        "their lighting over 2 x 2 or 4 x 4 pixels");
        mParameters.Add("volumeShareBelow2", &march.shareBelow2,
                        "Real-time volumes: a wave's transmittance below which its 2 x 2 pixels share lighting", 0.0f,
                        1.0f);
        mParameters.Add("volumeShareBelow4", &march.shareBelow4,
                        "Real-time volumes: a wave's transmittance below which its 4 x 4 pixels share lighting", 0.0f,
                        1.0f);
        mParameters.Add("volumeShareFootprint2", &march.shareFootprint2,
                        "Real-time volumes: pixels a voxel covers from which 2 x 2 pixels share lighting", 0.5f,
                        1000.0f);
        mParameters.Add("volumeShareFootprint4", &march.shareFootprint4,
                        "Real-time volumes: pixels a voxel covers from which 4 x 4 pixels share lighting", 0.5f,
                        1000.0f);
        mParameters.Add("volumeAir", &march.air,
                        "Real-time volumes: the aerial perspective dims them at the depth their light comes from");
        mParameters.Add(
            "volumeShadowsEveryFrame", &march.everyFrame,
            "Real-time volumes: render the shadow maps every frame (to time them), not only when they change");
        mParameters.Add("volumeTemporal", &march.temporal,
                        "Real-time volumes: filter their light over frames when no denoiser of the whole frame does");
        mParameters.Add("volumeTemporalBlend", &march.temporalBlend,
                        "Real-time volumes: the weight of a frame in that filter", 0.01f, 1.0f);
        mParameters.Add("furnace", &mSettings.furnace,
                        "White furnace: white materials, lights off; energy conservation shows as a flat grey image");
        mParameters.Add("debugView", &mSettings.debugView,
                        "0 lit, 1 normals, 2 base color, 3 roughness, 4 metallic, 5 depth, 6 intersection heatmap, "
                        "7 bounce heatmap, 8 direct light, 9 indirect light, 10 radiance cache at the first hit, "
                        "11 volume march steps, 12 its lighting rate, 13 volumes per ray",
                        0, gpu::kDebugViewCount - 1);
        mParameters.Add(
            "ser", &mSettings.shaderExecutionReordering,
            "Shader execution reordering (DXR 1.2) where the GPU reorders; slower with today's cheap shading");
        mParameters.Add("coatModel", &mSettings.coatModel,
                        "Clear coats: 0 neural (an MLP trained for the scene's coated materials), 1 random walk "
                        "(the reference, noisy), 2 analytic approximation",
                        0, kCoatModelApproximate);
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
        mTracer->SetTaskSystem(&context.tasks);
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
        // The guides cost about a tenth of the trace: only write them for a consumer (a denoiser, TAA).
        namespace slots = renderer::slots;
        live.guides =
            mSettings.guides && (context.Consumes(slots::kDiffuseAlbedo) || context.Consumes(slots::kSpecularAlbedo) ||
                                 context.Consumes(slots::kNormalRoughness) || context.Consumes(slots::kViewDepth) ||
                                 context.Consumes(slots::kMotionVectors));
        live.signals = live.guides && context.Consumes(slots::kDiffuseSignal);
        live.transparencyGuides = live.guides && context.Consumes(slots::kColorBeforeTransparency);
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
        status["neuralKernels"] = mTracer && mTracer->UsesNeuralKernels(); // AIR_PT_NEURAL
        status["neuralCoatLoss"] = mTracer ? mTracer->GetNeuralCoatLoss() : 0.0f;
        status["nrc"] = mTracer && mTracer->UsesRadianceCache();
        status["nrcLinearAlgebra"] = mTracer && mTracer->RadianceCacheUsesLinearAlgebra();
        status["nrcSteps"] = mTracer ? mTracer->GetRadianceCacheSteps() : 0u;
        status["volumeMarch"] = mTracer && mTracer->UsesVolumeMarch();
        status["volumeMarchLights"] = mTracer ? mTracer->GetVolumeMarchLights() : 0u;
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
