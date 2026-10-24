#pragma once

// CPU reference path tracer. Runs the GPU estimator (shaders/Integrator.hlsli) on the CPU with a
// simple BVH, so a GPU image can be checked against it pixel by pixel with `aircli compare`, and
// tests can verify energy conservation and NEE/MIS without a GPU. Slow by design: small images.

#include "air/core/Image.h"
#include "air/core/TaskSystem.h"
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
    bool furnace = false;
    // true: radiance * camera exposure, comparable with the pathtracer feature's output.
    // false: radiance in nits.
    bool applyExposure = true;
    TaskSystem* tasks = nullptr; // rows in parallel when set
};

Image RenderReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                      const ReferenceSettings& settings);

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
