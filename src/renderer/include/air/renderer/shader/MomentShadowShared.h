// Moment shadow maps of volumes (docs/architecture.md §8.13), shared by C++ and HLSL (see
// HlslCompat.h). A map stores, per texel, how the volumes' optical depth is spread along the light's
// ray: its total b0 and its power moments b1..b4 over a depth z in [-1, 1] (MomentShadow.hlsli), in
// an R16_FLOAT and an RGBA16_FLOAT slice of two texture arrays. A lookup anywhere (in a volume, on a
// surface) is one bilinear fetch and a closed-form reconstruction instead of a ray march towards the
// light: the shadow maps of translucent occluders of moment shadow mapping (Peters and Klein 2015;
// Peters et al. 2017) with the reconstruction of moment-based order-independent transparency
// (Muenstermann et al. 2018), rendered once per light for volumes as Mueller describes (SIGGRAPH 2026).
// A third R32_FLOAT slice holds the depth of the texel ray's first opaque hit (a plain shadow map,
// ray traced): the light that reaches a point inside a volume is that of both maps.
// A large emitter close to the volumes has several point maps, from points spread over it: the light
// it sends through them is the mean over its extent, not what its centre would see.
#ifndef AIR_RENDERER_MOMENT_SHADOW_SHARED_H
#define AIR_RENDERER_MOMENT_SHADOW_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kMomentMapSun = 0;   // orthographic, along the sun's light, over the volumes' box
AIR_CONST uint kMomentMapPoint = 1; // octahedral around a point (a light's centre)

struct MomentShadowMap
{
    float3 origin;     // sun: the corner of the map on its near plane; point: the centre
    uint kind;         // kMomentMapSun or kMomentMapPoint
    float3 axisU;      // sun: the map's first side (metres, perpendicular to axisW)
    float depthNear;   // the distance along a texel's ray where z = -1 (sun: from the near plane)
    float3 axisV;      // sun: its second side
    float depthFar;    // where z = 1
    float3 axisW;      // sun: the direction its light travels (unit)
    uint slice;        // in the map texture arrays
    uint size;         // texels per side
    uint skipInstance; // point: the light's own instance, which its rays do not hit (kInvalidIndex: none)
    uint count;        // the light's maps: this one and the count - 1 after it
    uint pad0;
};

// What a map's opaque slice holds where the texel's ray hits nothing, and for a sun map whose texel
// ray is blocked before it reaches the volumes' box.
AIR_CONST float kMomentNoOccluder = 2.0f;
AIR_CONST float kMomentOccluded = -2.0f;

#ifdef __cplusplus
static_assert(sizeof(MomentShadowMap) == 80);
} // namespace air::gpu
#endif

#endif
