// The sky as the renderers see it, shared by C++ and HLSL (see HlslCompat.h): a table of sky
// radiance over all directions (renderer::BuildSkyTable writes it, docs/architecture.md §8.9) and
// the sun as a disc light. SkyLookup.hlsli reads the table.
//
// Table layout: u = sqrt(phi / pi) for the azimuth phi between the direction and the sun (the sky
// is mirror-symmetric about the sun's vertical plane), v = 0.5 + 0.5 sign(e) sqrt(|e| / (pi / 2)) for
// the elevation e. Both square roots spend the texels near the sun's azimuth and near the horizon.
//
// Aerial perspective: two RGBA16F 3D textures over the same (u, v), at a lower resolution, and the
// distance: slice k holds aerialDistance * (k / (aerialSlices - 1))^2. One holds the radiance the air
// scatters towards the viewer over that distance (nits, divided by kAerialInscatterScale), the other its
// transmittance.
#ifndef AIR_RENDERER_SKY_SHARED_H
#define AIR_RENDERER_SKY_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

struct SkyData
{
    float3 sunDirection; // unit, towards the sun (y up)
    float sunCosRadius;  // cosine of its angular radius
    float3 sunRadiance;  // nits, after the atmosphere; 0: no sun
    uint width;          // of the radiance table; 0: a uniform sky of radiance `uniformRadiance`
    float3 uniformRadiance;
    uint height;
    float aerialDistance;     // metres the aerial perspective reaches; 0: no air in front of anything
    uint aerialSlices;        // distance slices of its tables
    uint aerialInscatter;     // bindless Texture3D<float4> (GPU only)
    uint aerialTransmittance; // bindless Texture3D<float4> (GPU only)
};

#ifdef __cplusplus
static_assert(sizeof(SkyData) == 64);
#endif

AIR_CONST float kSkyPi = 3.14159265358979f;
// The inscatter table holds nits over this: towards a low sun the air scatters past half precision's
// 65504 nits over tens of kilometres (an infinite texel, then NaN wherever the clouds' light went
// through it); scaled, the table reaches 1.7e7 nits and its faintest texels keep their precision.
AIR_CONST float kAerialInscatterScale = 256.0f;

// Table coordinates (0..1) of a direction.
AIR_INLINE float2 SkyTableUv(float3 direction, float3 sunDirection)
{
    float elevation = asin(clamp(direction.y, -1.0f, 1.0f));
    float sunHorizontal = sunDirection.x * sunDirection.x + sunDirection.z * sunDirection.z;
    float horizontal = direction.x * direction.x + direction.z * direction.z;
    float cosPhi = 1.0f;
    if (sunHorizontal > 1e-12f && horizontal > 1e-12f)
    {
        cosPhi = (direction.x * sunDirection.x + direction.z * sunDirection.z) * rsqrt(sunHorizontal * horizontal);
    }
    float phi = acos(clamp(cosPhi, -1.0f, 1.0f));
    float v = 0.5f + 0.5f * (elevation < 0.0f ? -1.0f : 1.0f) * sqrt(abs(elevation) / (0.5f * kSkyPi));
    return float2(sqrt(phi / kSkyPi), v);
}

#ifdef __cplusplus
} // namespace air::gpu
#endif

#endif
