// ReSTIR data shared by C++ and HLSL (see air/renderer/shader/HlslCompat.h, ReSTIRDI.hlsli and
// ReSTIRGI.hlsli).
#ifndef AIR_PT_RESTIR_SHARED_H
#define AIR_PT_RESTIR_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kReSTIRUnbiased = 1u; // trace the visibility the MIS weights need (else a darkening bias)
AIR_CONST uint kReSTIRTemporal = 2u; // last frame's primary hits and reservoirs are valid
AIR_CONST uint kReSTIRShade = 4u;    // the spatial pass adds its estimate to the accumulation
AIR_CONST uint kReSTIRMaxSpatialSamples = 16u;
AIR_CONST uint kReSTIRMaxLight = 0xFFFFFFu; // light indices have 24 bits in a reservoir

// The first surface a pixel's camera ray hit, as PathTracer.Trace found it (its first sample). The
// ReSTIR passes rebuild the surface from it with the integrator's LoadSurface, so they shade
// exactly what the path tracer shades; normal, roughness and distance serve the choice of
// neighbours.
struct PrimaryHit
{
    uint instance; // kInvalidIndex: the camera ray left the scene
    uint primitive;
    float2 barycentrics;
    uint normalAndRoughness; // shading normal, octahedral 12 + 12 bits snorm; roughness in bits 24-31
    float distance;          // from the eye
};

// One light sample and its resampling state: a point on an emissive triangle (light-list entry and
// barycentrics), its unbiased contribution weight W and its confidence M (candidate count).
struct Reservoir
{
    uint lightAndConfidence; // light index in bits 0-23, M (0..255) in bits 24-31
    float2 barycentrics;
    float weight; // W; 0: no sample (none found, or occluded where it was found)
};

// A ReSTIR GI sample (Integrator.hlsli, GiPath): the path beyond its second vertex x2, which
// shifts to another pixel by reconnecting that pixel's first surface to x2.
struct GiReservoir
{
    uint instance; // x2; kInvalidIndex: no sample
    uint primitive;
    float2 barycentrics;
    float3 direction; // at x2, towards the rest of the path
    float lightPdf;   // solid angle pdf of NEE at x2 choosing the light at the end of `direction`
    float3 radianceMis;
    float weight; // W
    float3 radianceFixed;
    uint strategyAndConfidence; // strategy (0 NEE at x2, 1 BSDF continuation) in bit 0, M in bits 8-15
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
    uint input;               // StructuredBuffer of reservoirs: initial (temporal pass) or temporal (spatial)
    uint previous;            // StructuredBuffer of reservoirs: last frame's final ones
    uint output;              // RWStructuredBuffer of reservoirs (Reservoir or GiReservoir)
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
static_assert(sizeof(GiReservoir) == 64);
static_assert(sizeof(ReSTIRConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pt::gpu
#else
// PrimaryHit::normalAndRoughness: a unit vector in the octahedral map (Cigolle et al. 2014), 12-bit
// snorm per axis, and a roughness in the top 8 bits.
uint EncodeNormalRoughness(float3 n, float roughness)
{
    float2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0f)
    {
        p = (1.0f - abs(p.yx)) * select(p >= 0.0f, 1.0f, -1.0f);
    }
    int2 q = int2(round(clamp(p, -1.0f, 1.0f) * 2047.0f));
    return (uint(q.x) & 0xFFFu) | ((uint(q.y) & 0xFFFu) << 12) | (uint(round(saturate(roughness) * 255.0f)) << 24);
}

float DecodeRoughness(uint packed)
{
    return float(packed >> 24) / 255.0f;
}

float3 DecodeNormal(uint packed)
{
    float2 p = float2(int2(int(packed << 20) >> 20, int(packed << 8) >> 20)) / 2047.0f;
    float3 n = float3(p, 1.0f - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0f, -t, t);
    return normalize(n);
}
#endif

#endif
