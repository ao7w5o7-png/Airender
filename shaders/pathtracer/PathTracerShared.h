// Shared between C++ and HLSL. Keep every struct 16-byte aligned and layout-identical on both sides.
#ifndef AIR_PATH_TRACER_SHARED_H
#define AIR_PATH_TRACER_SHARED_H

#ifdef __cplusplus
#include "air/core/Types.h"
namespace air::pt {
using uint = air::u32;
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
#define AIR_PT_ALIGN alignas(16)
#else
#define AIR_PT_ALIGN
#endif

// Root constants / push constants for the trace pass. All resources are bindless indices.
struct AIR_PT_ALIGN TraceConstants {
    uint tlas;            // RaytracingAccelerationStructure
    uint accumulationUav; // RWTexture2D<float4>: rgb = radiance sum, a = sample count
    uint instanceBuffer;  // StructuredBuffer<InstanceData>
    uint materialBuffer;  // StructuredBuffer<MaterialData>

    uint frameIndex;      // seeds the RNG
    uint samplesPerPixel; // samples traced this dispatch
    uint maxBounces;
    uint resetAccumulation;

    float4 cameraPositionFov; // xyz = position, w = vertical fov (radians)
    float4 cameraForward;     // xyz
    float4 cameraRight;       // xyz
    float4 cameraUp;          // xyz
};

struct AIR_PT_ALIGN ResolveConstants {
    uint accumulationSrv;
    uint outputUav;
    float exposure;
    uint width;
    uint height;
    uint pad0, pad1, pad2;
};

struct AIR_PT_ALIGN InstanceData {
    uint vertexBuffer;  // StructuredBuffer<Vertex>
    uint indexBuffer;   // StructuredBuffer<uint>
    uint materialIndex;
    uint pad;
};

struct AIR_PT_ALIGN Vertex {
    float3 position;
    float u;
    float3 normal;
    float v;
};

struct AIR_PT_ALIGN MaterialData {
    float3 baseColor;
    float roughness;
    float3 emission;
    float metallic;
};

#ifdef __cplusplus
static_assert(sizeof(TraceConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(ResolveConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(Vertex) == 32);
static_assert(sizeof(MaterialData) == 32);
} // namespace air::pt
#endif

#endif
