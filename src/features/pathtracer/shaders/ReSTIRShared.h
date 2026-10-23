// ReSTIR DI data shared by C++ and HLSL (see air/renderer/shader/HlslCompat.h and ReSTIR.hlsli).
#ifndef AIR_PT_RESTIR_SHARED_H
#define AIR_PT_RESTIR_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kReSTIRUnbiased = 1u;  // trace the visibility the MIS weights need (else a darkening bias)
AIR_CONST uint kReSTIRTemporal = 2u;  // last frame's primary hits and reservoirs are valid
AIR_CONST uint kReSTIRAddDirect = 4u; // the spatial pass adds the direct light to the accumulation
AIR_CONST uint kReSTIRMaxSpatialSamples = 16u;
AIR_CONST uint kReSTIRMaxLight = 0xFFFFFFu; // light indices have 24 bits in a reservoir

// The first surface a pixel's camera ray hit, as PathTracer.Trace found it (its first sample). The
// ReSTIR passes rebuild the surface from it with the integrator's LoadSurface, so they shade
// exactly what the path tracer shades; normal and distance serve the neighbour similarity tests.
struct PrimaryHit
{
    uint instance; // kInvalidIndex: the camera ray left the scene
    uint primitive;
    float2 barycentrics;
    uint normal;    // shading normal, octahedral, 16 + 16 bits snorm
    float distance; // from the eye
};

// One light sample and its resampling state: a point on an emissive triangle (light-list entry and
// barycentrics), its unbiased contribution weight W and its confidence M (candidate count).
struct Reservoir
{
    uint lightAndConfidence; // light index in bits 0-23, M (0..255) in bits 24-31
    float2 barycentrics;
    float weight; // W; 0: no sample (none found, or occluded where it was found)
};

// Root constants of the ReSTIR passes: the trace pass's scene indices and the pass's buffers.
struct ReSTIRConstants
{
    uint frameData; // StructuredBuffer<FrameData>
    uint frameSlot;
    uint tlas;
    uint instances;
    uint materials;
    uint lights;
    uint instanceLights;
    uint lightTree;
    uint bsdfLut;
    uint primaryHits;         // StructuredBuffer<PrimaryHit>, this frame
    uint previousPrimaryHits; // StructuredBuffer<PrimaryHit>, last frame
    uint input;               // StructuredBuffer<Reservoir>: initial (temporal pass) or temporal (spatial)
    uint previous;            // StructuredBuffer<Reservoir>: last frame's final reservoirs
    uint output;              // RWStructuredBuffer<Reservoir>
    uint accumulation;        // RWTexture2D<float4>: the spatial pass adds the direct light
    uint flags;               // kReSTIR*
    uint lightCandidates;     // initial RIS: light samples chosen by power
    uint bsdfCandidates;      // initial RIS: BSDF rays that may hit an emitter
    uint spatialSamples;      // neighbours the spatial pass reuses
    uint disocclusionSamples; // neighbours where there is no temporal history
    float spatialRadius;      // pixels
    float maxConfidence;      // cap of the temporal history's M
    uint pad0;
    uint pad1;
};

#ifdef __cplusplus
static_assert(sizeof(PrimaryHit) == 24);
static_assert(sizeof(Reservoir) == 16);
static_assert(sizeof(ReSTIRConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pt::gpu
#else
// Unit vector <-> octahedral map (Cigolle et al. 2014), 16-bit snorm per axis (PrimaryHit::normal).
uint EncodeOctahedral(float3 n)
{
    float2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0f)
    {
        p = (1.0f - abs(p.yx)) * select(p >= 0.0f, 1.0f, -1.0f);
    }
    int2 q = int2(round(clamp(p, -1.0f, 1.0f) * 32767.0f));
    return (uint(q.x) & 0xFFFFu) | (uint(q.y) << 16);
}

float3 DecodeOctahedral(uint packed)
{
    float2 p = float2(int2(int(packed << 16) >> 16, int(packed) >> 16)) / 32767.0f;
    float3 n = float3(p, 1.0f - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0f, -t, t);
    return normalize(n);
}
#endif

#endif
