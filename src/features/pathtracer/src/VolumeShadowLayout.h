#pragma once

// Which lights shadow the real-time volumes and where their moment shadow maps lie (docs/architecture.md
// §8.13), for the GPU (VolumeMarcher) and its CPU twin (VolumeMarchReference.cpp) alike: the light
// groups that matter most to the volumes, a map of the sun over the volumes' box and around each of
// those groups one map, or for a large emitter close to the volumes up to kMaxLightOrigins of them
// from points spread over it.

#include "air/features/pathtracer/PathTracer.h"
#include "air/renderer/shader/SceneShared.h"
#include "air/renderer/shader/SkyShared.h"

#include "VolumeMarchShared.h"

#include <span>
#include <vector>

namespace air::pt
{

struct VolumeShadowLayout
{
    std::vector<air::gpu::MomentShadowMap> maps; // map i in slice i, the sun's first
    std::vector<u32> groupMaps; // per light group (at least one entry): its first map, kInvalidIndex: none
    std::vector<u32> lights;    // the light groups with maps, the one that matters most first
    u32 sunMap = air::gpu::kInvalidIndex;
};

// The settings within the ranges the passes support.
VolumeMarchSettings ClampVolumeMarchSettings(const VolumeMarchSettings& settings);

// Up to settings.lights light groups (`lights` holds their triangles), by power over their squared
// distance to the volumes' box, and the maps of settings.shadowSize texels per side: orthographic for
// the sun when it shines, octahedral around points of each group (its centre, or with
// settings.lightOrigins > 1 the centres of the four quadrants of a large open emitter close to the
// volumes), their depths spanning the box as seen from there.
VolumeShadowLayout LayoutVolumeShadows(std::span<const air::gpu::VolumeMarchData> volumes,
                                       std::span<const air::gpu::LightGroup> groups,
                                       std::span<const air::gpu::LightTriangle> lights, const air::gpu::SkyData& sky,
                                       const VolumeMarchSettings& settings);

} // namespace air::pt
