#pragma once

// CPU reference path tracer. Runs the GPU estimator (shaders/Integrator.hlsli) on the CPU with a
// simple BVH, so a GPU image can be checked against it pixel by pixel with `aircli compare`, and
// tests can verify energy conservation and NEE/MIS without a GPU. Slow by design: small images.

#include "air/core/Image.h"
#include "air/core/TaskSystem.h"
#include "air/renderer/Scene.h"

namespace air::pt {

struct ReferenceSettings {
    u32 width = 64;
    u32 height = 64;
    u32 samplesPerPixel = 64;
    u32 maxBounces = 8;
    bool nextEventEstimation = true;
    bool furnace = false;
    // true: radiance * camera exposure, comparable with the pathtracer feature's output.
    // false: radiance in nits.
    bool applyExposure = true;
    TaskSystem* tasks = nullptr; // rows in parallel when set
};

Image RenderReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                      const ReferenceSettings& settings);

} // namespace air::pt
