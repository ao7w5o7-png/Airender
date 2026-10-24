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
    uint reserved; // the light-list offset changes with materials: GpuScene::instanceLights
    // Row-major 3x4 object-to-world transform and the 3x3 inverse transpose for normals.
    float4 objectToWorld0;
    float4 objectToWorld1;
    float4 objectToWorld2;
    float4 normalToWorld0;
    float4 normalToWorld1;
    float4 normalToWorld2;
};

// An emissive triangle in world space. Next event estimation chooses one of two ways:
//   * by power alone: the list doubles as a Walker/Vose alias table, so choosing is O(1): draw
//     entry i uniformly, keep it if a second uniform is below aliasThreshold, else take `alias`;
//   * by the light tree (LightTreeNode), which also weighs distance and orientation from the
//     shading point. treeTrail is the path from the root to the light's leaf (bit d set: the second
//     child at depth d), to recompute the probability the tree gives the light (for MIS).
struct LightTriangle
{
    float3 p0;
    float area;
    float3 e1;         // p1 - p0; the emitting side is cross(e1, e2)
    float probability; // of being chosen by power
    float3 e2;         // p2 - p0
    float aliasThreshold;
    float3 radiance;
    uint alias;
    uint treeTrailLow;  // trail bits 0-31
    uint treeTrailHigh; // trail bits 32-63
    uint treeDepth;     // depth of its leaf; kInvalidIndex: not in the tree (no power)
    uint pad0;
};

// A node of the light tree (renderer::BuildLightTree) over the emissive triangles: it bounds the
// emitters below it in space (a box), in emission direction (normals within cosThetaO of `axis`,
// each emitting within cosThetaE of its normal) and in power. Sampling walks down from the root and
// picks each child with probability proportional to its importance at the shading point (power x
// orientation bound / distance^2; Conty Estevez and Kulla 2018). Internal node: the first child
// follows it, the second is at `child`. Leaf: `child` is the light.
struct LightTreeNode
{
    float3 boundsMin;
    float power; // of the emitters below, relative
    float3 boundsMax;
    float cosThetaO;
    float3 axis;
    float cosThetaE;
    uint child;
    uint isLeaf;
    uint pad0;
    uint pad1;
};

#ifdef __cplusplus
static_assert(sizeof(Vertex) == 32);
static_assert(sizeof(MaterialData) == 32);
static_assert(sizeof(InstanceData) == 112);
static_assert(sizeof(LightTriangle) == 80);
static_assert(sizeof(LightTreeNode) == 64);
} // namespace air::gpu
#endif

#endif
