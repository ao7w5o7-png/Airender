#pragma once

// CPU reference path tracer. Runs the GPU estimator (shaders/Integrator.hlsli) on the CPU with a
// simple BVH, so a GPU image can be checked against it pixel by pixel with `aircli compare`, and
// tests can verify energy conservation and NEE/MIS without a GPU. Slow by design: small images.

#include "air/core/Image.h"
#include "air/core/TaskSystem.h"
#include "air/features/pathtracer/PathTracer.h"
#include "air/renderer/Scene.h"

#include <memory>

namespace air::pt
{

struct ReferenceSettings
{
    u32 width = 64;
    u32 height = 64;
    u32 samplesPerPixel = 64;
    u32 maxBounces = 8;
    bool nextEventEstimation = true;
    u32 lightTreeBounces = 0; // as pathtracer.lightTreeBounces
    u32 cloudEvents = 256;    // as pathtracer.cloudEvents
    u32 volumeEvents = 256;   // as pathtracer.volumeEvents
    u32 coatModel = 1;        // Bsdf.hlsli kCoat* (1 neural, 2 random walk, 3 approximate): pathtracer.coatModel + 1
    bool furnace = false;
    // true: radiance * camera exposure, comparable with the pathtracer feature's output.
    // false: radiance in nits.
    bool applyExposure = true;
    TaskSystem* tasks = nullptr; // rows in parallel when set
};

Image RenderReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                      const ReferenceSettings& settings);

// The real-time volumes (pathtracer.volumeMarch, docs/architecture.md §8.13) on the CPU, with the
// GPU's shader code: their moment shadow maps, their probes (the light of probeFrames frames of them
// averaged, as when accumulating), the surfaces path traced and lit through the maps, and the ray march
// over them. Frame after frame like the pathtracer feature accumulating with pathtracer.volumeMarch=2:
// one camera ray per pixel at the frame's sub-pixel offset and march jitter, settings.samplesPerPixel
// paths along it. The march lights every step (pathtracer.volumeAdaptive=0). For checking the method
// against RenderReference, which tracks the volumes, and the GPU against the method.
struct VolumeMarchReferenceSettings
{
    u32 frames = 16;
    u32 probeFrames = 32;
    VolumeMarchSettings march; // its step, level, shadowSize, shadowLevel, shadowStep, lights and air
};

Image RenderVolumeMarchReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                                 const ReferenceSettings& settings, const VolumeMarchReferenceSettings& march);

// The real-time volumes' lighting on the CPU, for tests: their moment shadow maps (VolumeShadows.hlsl)
// and what the sun's lets through to a point (as surfaces read it, MomentShadow) and inside the
// volumes (with the geometry's shadow, MomentLightVisibility), against the transmittance marched
// finely through the voxels; and the light a light group sends through a point of the march.
class VolumeMarchProbe
{
public:
    VolumeMarchProbe(const renderer::SceneDesc& scene, const VolumeMarchSettings& settings,
                     TaskSystem* tasks = nullptr);
    ~VolumeMarchProbe();
    VolumeMarchProbe(const VolumeMarchProbe&) = delete;
    VolumeMarchProbe& operator=(const VolumeMarchProbe&) = delete;

    u32 GetMapCount() const;
    u32 GetLightCount() const; // light groups with maps
    bool HasSunMap() const;
    float SunShadow(renderer::float3 p) const;
    float SunVisibility(renderer::float3 p) const;
    float SunTransmittance(renderer::float3 p) const; // the volumes' alone, marched at a quarter voxel
    // The same for light group `group` (the mean over its maps, MomentLightVisibility, as the frames
    // read them in turn; 1 without maps) and towards the points they look from.
    float GroupVisibility(u32 group, renderer::float3 p) const;
    float GroupTransmittance(u32 group, renderer::float3 p) const;
    u32 GetGroupMapCount(u32 group) const;
    // The light of light group `group` (one per emissive instance, in their order) that a medium at x
    // scatters along direction d, per unit scattering coefficient, with a Henyey-Greenstein phase
    // function of `anisotropy`, before shadows (VolumeMarch.hlsli's GroupInScattering).
    renderer::float3 GroupInScattering(u32 group, renderer::float3 x, renderer::float3 d, float anisotropy) const;
    // The direct light (the sun's and the light groups', through map `sample` of each) that the volumes
    // at x scatter along direction d per unit length: as a step of the march lights itself
    // (MarchDirect), and as a pixel lights it for its neighbours under adaptive sampling, per unit of
    // scattering (MarchLightAt) times the scattering at x; and that scattering.
    renderer::float3 StepLight(renderer::float3 x, renderer::float3 d, u32 sample) const;
    renderer::float3 SharedStepLight(renderer::float3 x, renderer::float3 d, u32 sample) const;
    float Scattering(renderer::float3 x) const;

private:
    struct State;
    std::unique_ptr<State> mState;
};

// The integrator's light tree functions (Integrator.hlsli, shared with the GPU) on a scene's lights,
// for tests: the probability next event estimation at point p with normal n chooses a light, and
// the light it chooses with a uniform u.
class LightTreeProbe
{
public:
    explicit LightTreeProbe(const renderer::SceneDesc& scene);
    ~LightTreeProbe();
    LightTreeProbe(const LightTreeProbe&) = delete;
    LightTreeProbe& operator=(const LightTreeProbe&) = delete;

    u32 GetLightCount() const;
    u32 GetNodeCount() const;
    u32 GetDepth(u32 light) const; // ~0u: not in the tree
    float Pmf(u32 light, renderer::float3 p, renderer::float3 n) const;
    u32 Sample(renderer::float3 p, renderer::float3 n, float u, float* pmf) const; // ~0u: none

private:
    struct Tables;
    std::unique_ptr<Tables> mTables;
};

} // namespace air::pt
