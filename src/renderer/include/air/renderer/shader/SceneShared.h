// GPU scene layout shared by C++ and HLSL (see HlslCompat.h). Every struct here is the element of
// a StructuredBuffer, so float3 + scalar packs tightly into 16 bytes on both sides.
//
// Units: lengths in metres, emitted radiance in nits (cd/m^2, the luminance of `emission`).
#ifndef AIR_RENDERER_SCENE_SHARED_H
#define AIR_RENDERER_SCENE_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kInvalidIndex = 0xFFFFFFFFu;

struct Vertex
{
    float3 position; // object space
    float u;
    float3 normal; // object space, normalized
    float v;
};

struct MaterialData
{
    float3 baseColor; // linear albedo (dielectric) or F0 (metal)
    float roughness;  // perceptual
    float3 emission;  // emitted radiance, nits
    float metallic;
};

// One TLAS instance. InstanceID() indexes this array.
struct InstanceData
{
    uint vertexBuffer; // bindless StructuredBuffer<Vertex>
    uint indexBuffer;  // bindless StructuredBuffer<uint>
    uint materialIndex;
    uint firstLight; // index of triangle 0 in the light list, kInvalidIndex when not emissive
    // Row-major 3x4 object-to-world transform and the 3x3 inverse transpose for normals.
    float4 objectToWorld0;
    float4 objectToWorld1;
    float4 objectToWorld2;
    float4 normalToWorld0;
    float4 normalToWorld1;
    float4 normalToWorld2;
};

// An emissive triangle in world space, chosen for next event estimation proportionally to power.
struct LightTriangle
{
    float3 p0;
    float area;
    float3 e1;         // p1 - p0; the emitting side is cross(e1, e2)
    float probability; // of being chosen
    float3 e2;         // p2 - p0
    float cdf;         // inclusive prefix sum of probability
    float3 radiance;
    uint instance;
};

#ifdef __cplusplus
static_assert(sizeof(Vertex) == 32);
static_assert(sizeof(MaterialData) == 32);
static_assert(sizeof(InstanceData) == 112);
static_assert(sizeof(LightTriangle) == 64);
} // namespace air::gpu
#endif

#endif
