// Shared by the ReSTIR passes (ReSTIRDI.hlsli, ReSTIRGI.hlsli): their push constants, the
// integrator compiled with inline ray queries, each pixel's first surface rebuilt from its
// PrimaryHit (a Receiver, exactly as TracePath shades it), neighbour selection by geometry, and
// weighted reservoir sampling.
#ifndef AIR_PT_RESTIR_COMMON_HLSLI
#define AIR_PT_RESTIR_COMMON_HLSLI

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"
#include "ReSTIRShared.h"

[[vk::push_constant]] ConstantBuffer<ReSTIRConstants> g : register(b0);

#include "SceneAccess.hlsli"

HitInfo TraceClosest(float3 origin, float3 direction, uint bounce)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    HitInfo hit;
    hit.hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    hit.t = hit.hit ? query.CommittedRayT() : -1.0f;
    hit.instance = hit.hit ? query.CommittedInstanceID() : 0u;
    hit.primitive = hit.hit ? query.CommittedPrimitiveIndex() : 0u;
    hit.barycentrics = hit.hit ? query.CommittedTriangleBarycentrics() : float2(0.0f, 0.0f);
    return hit;
}

#include "Integrator.hlsli"

AIR_CONST float kNormalThreshold = 0.5f;   // cosine between the normals of surfaces that share samples
AIR_CONST float kDistanceThreshold = 0.1f; // relative difference of their distances to the eye

FrameData LoadFrame()
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    return frames[g.frameSlot];
}

IntegratorParams MakeParams(FrameData f)
{
    IntegratorParams params;
    params.sky = f.sky;
    params.clouds = f.clouds;
    params.volumeCount = f.volumeCount;
    params.volumeTable = f.volumeTable;
    params.lightCount = f.lightCount;
    params.maxBounces = f.maxBounces;
    params.flags = f.flags;
    params.lightTreeBounces = f.lightTreeBounces;
    params.cloudEvents = f.cloudEvents;
    params.volumeEvents = f.volumeEvents;
    params.nrcSpread = 0.0f; // ReSTIR's paths do not end in the radiance cache
    params.nrcSlot = 0u;
    params.nrcRecords = 0u;
    return params;
}

// What the air between the camera and a first surface lets through: the paths' own light there gets
// the same (AddAerialPerspective, which also adds the air's light).
float3 CameraAirTransmittance(FrameData f, float3 position)
{
    float3 toPoint = position - f.camera.position.xyz;
    float distance = length(toPoint);
    return AerialPerspective(f.sky, toPoint / max(distance, 1e-6f), distance).transmittance;
}

// Independent random streams per pixel, frame and use.
uint StreamSeed(uint2 pixel, uint frameIndex, uint stream)
{
    return Pcg(pixel.x + Pcg(pixel.y + Pcg(frameIndex + Pcg(stream + 0x52455354u))));
}

PrimaryHit LoadPrimaryHit(uint buffer, uint index)
{
    StructuredBuffer<PrimaryHit> hits = ResourceDescriptorHeap[buffer];
    return hits[index];
}

// Whether a surface (b) is close enough to another (a) in normal and distance to the same eye to
// share samples with it. Only a heuristic for variance: the MIS weights keep any reuse unbiased.
bool SimilarSurface(PrimaryHit a, PrimaryHit b, float expectedDistance)
{
    return b.instance != kInvalidIndex &&
           dot(DecodeNormal(a.normalAndRoughness), DecodeNormal(b.normalAndRoughness)) >= kNormalThreshold &&
           abs(b.distance - expectedDistance) <= kDistanceThreshold * expectedDistance;
}

// --- Receivers and light samples -----------------------------------------------------------------

// A pixel's first surface as TracePath shades it at bounce 0, rebuilt from its PrimaryHit and seen
// from `eye`.
struct Receiver
{
    bool valid;
    float3 position;
    float3 origin; // where its rays start, off the surface
    float3 ng;     // geometric normal, facing the eye
    Frame frame;   // around the shading normal the BSDF uses
    BsdfContext bsdf;
};

Receiver LoadReceiver(PrimaryHit h, float3 eye, IntegratorParams params)
{
    Receiver r = (Receiver)0;
    r.valid = false;
    if (h.instance == kInvalidIndex)
    {
        return r;
    }
    HitInfo hit;
    hit.hit = true;
    hit.t = h.distance;
    hit.instance = h.instance;
    hit.primitive = h.primitive;
    hit.barycentrics = h.barycentrics;
    // LoadSurface orients the normals against a ray, unknown here: orient them towards the eye.
    Surface s = LoadSurface(hit, float3(0.0f, 0.0f, 0.0f), params);
    float3 toEye = eye - s.position;
    if (dot(s.geometricNormal, toEye) < 0.0f)
    {
        s.geometricNormal = -s.geometricNormal;
        s.shadingNormal = -s.shadingNormal;
    }
    float3 wo = normalize(toEye);
    float3 ns = dot(s.shadingNormal, wo) > 0.0f ? s.shadingNormal : s.geometricNormal;
    r.valid = true;
    r.position = s.position;
    r.origin = OffsetRayOrigin(s.position, s.geometricNormal);
    r.ng = s.geometricNormal;
    r.frame = MakeFrame(ns);
    r.bsdf = PrepareBsdf(s.material, ToLocal(r.frame, wo));
    return r;
}

// Draws the next spatial neighbour of `pixel`: uniform in the disc of spatialRadius, accepted when
// inside the image, similar to the centre and at least minRoughness rough. Replaying the same `draw`
// state gives the same neighbours.
bool NextNeighbour(inout uint draw, uint2 pixel, PrimaryHit center, FrameData f, float minRoughness,
                   out uint neighbour)
{
    float radius = g.spatialRadius * sqrt(Rand(draw));
    float angle = kTwoPi * Rand(draw);
    int2 q = int2(pixel) + int2(round(radius * float2(cos(angle), sin(angle))));
    neighbour = 0u;
    if (any(q < 0) || q.x >= int(f.width) || q.y >= int(f.height) || all(q == int2(pixel)))
    {
        return false;
    }
    neighbour = uint(q.y) * f.width + uint(q.x);
    PrimaryHit candidate = LoadPrimaryHit(g.primaryHits, neighbour);
    return SimilarSurface(center, candidate, center.distance) &&
           DecodeRoughness(candidate.normalAndRoughness) >= minRoughness;
}

#endif
