// Path tracer data shared by C++ and HLSL (see air/renderer/shader/HlslCompat.h).
#ifndef AIR_PATH_TRACER_SHARED_H
#define AIR_PATH_TRACER_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
#endif

// Debug views (pathtracer.debugView). Everything but kDebugNone replaces the radiance.
AIR_CONST uint kDebugNone = 0;
AIR_CONST uint kDebugNormals = 1;   // first-hit shading normal, n * 0.5 + 0.5
AIR_CONST uint kDebugBaseColor = 2; // first-hit base color
AIR_CONST uint kDebugRoughness = 3;
AIR_CONST uint kDebugMetallic = 4;
AIR_CONST uint kDebugDepth = 5;         // first-hit distance / debugScale
AIR_CONST uint kDebugIntersections = 6; // heatmap: triangles intersected by the camera ray / debugScale
AIR_CONST uint kDebugBounces = 7;       // heatmap: path length / maxBounces (Russian roulette shows here)
AIR_CONST uint kDebugDirect = 8;        // radiance of paths with at most one bounce (emission + direct light)
AIR_CONST uint kDebugIndirect = 9;      // radiance of the remaining paths
AIR_CONST uint kDebugViewCount = 10;

AIR_CONST uint kFlagFurnace = 1u;      // every material white, emitters off: an energy-conserving image is uniform
AIR_CONST uint kFlagReset = 2u;        // discard the accumulated samples
AIR_CONST uint kFlagGlobalJitter = 4u; // every sample uses FrameData.jitter (for temporal denoisers)
AIR_CONST uint kFlagGuides = 8u;       // write the denoiser guide textures
AIR_CONST uint kFlagNoNee = 16u;       // BSDF sampling only (reference for the MIS tests)
// The direct light at the first surface comes from ReSTIR DI (ReSTIR.hlsli): no NEE there, no
// emission found by its BSDF sample; the first sample's hit goes to TraceConstants::primaryHits.
AIR_CONST uint kFlagReSTIR = 32u;

// Result of TraceClosest() in the integrator (not stored in buffers).
struct HitInfo
{
    bool hit;
    float t;
    uint instance;  // InstanceID()
    uint primitive; // PrimitiveIndex()
    float2 barycentrics;
};

struct CameraData
{
    float4 position; // xyz, w = tan(verticalFov / 2)
    float4 forward;  // xyz, w = aspect (width / height)
    float4 right;
    float4 up;
};

// Everything a frame needs that does not fit in push constants. One element per frame in an
// upload ring; TraceConstants::frameSlot selects it.
struct FrameData
{
    CameraData camera;
    CameraData previousCamera; // for motion vectors
    float4 environment;        // rgb radiance (nits), w = exposure scale applied by Resolve
    float2 jitter;             // sub-pixel offset in [-0.5, 0.5) for kFlagGlobalJitter
    uint frameIndex;
    uint flags;
    uint width;
    uint height;
    uint samplesPerPixel;
    uint maxBounces;
    uint lightCount;
    uint debugView;
    float debugScale;      // normalizes depth / intersection debug views
    uint lightTreeBounces; // NEE uses the light tree at path vertices below this, power beyond
};

// Root constants of PathTracer.Trace: bindless indices only.
struct TraceConstants
{
    uint frameData; // StructuredBuffer<FrameData>
    uint frameSlot;
    uint tlas;
    uint instances;       // StructuredBuffer<InstanceData>
    uint materials;       // StructuredBuffer<MaterialData>
    uint lights;          // StructuredBuffer<LightTriangle>
    uint bsdfLut;         // StructuredBuffer<float>
    uint accumulation;    // RWTexture2D<float4>: rgb = sum, a = sample count
    uint diffuseAlbedo;   // RWTexture2D<float4>, written with kFlagGuides
    uint specularAlbedo;  // RWTexture2D<float4>
    uint normalRoughness; // RWTexture2D<float4>
    uint viewDepth;       // RWTexture2D<float>
    uint motionVectors;   // RWTexture2D<float2>
    uint instanceLights;  // StructuredBuffer<uint>: first light-list entry per instance
    uint lightTree;       // StructuredBuffer<LightTreeNode>
    uint primaryHits;     // RWStructuredBuffer<PrimaryHit>, written with kFlagReSTIR
};

struct ResolveConstants
{
    uint frameData;
    uint frameSlot;
    uint accumulation; // Texture2D<float4>
    uint output;       // RWTexture2D<float4>
};

#ifdef __cplusplus
static_assert(sizeof(CameraData) == 64);
static_assert(sizeof(FrameData) == 192);
static_assert(sizeof(TraceConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(ResolveConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pt::gpu
#endif

#endif
