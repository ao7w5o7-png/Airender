// GPU path tracer kernel: one pixel per thread, bindless resources. Compiled five times:
// PathTracer.hlsl (DXR 1.1, SM 6.6, every RTX GPU), PathTracerSer.hlsl (AIR_PT_SER: DXR 1.2 shader
// execution reordering, SM 6.9), PathTracerGi.hlsl (AIR_PT_RESTIR_GI: records ReSTIR GI's
// candidates), PathTracerNrc.hlsl and PathTracerNrcPortable.hlsl (AIR_PT_NRC: paths end in the
// neural radiance cache, evaluated with Shader Model 6.10 linear algebra or in float32);
// PathTracer.cpp picks one at runtime. The estimator itself is
// Integrator.hlsli, shared with the CPU reference renderer. This file supplies TraceClosest (an inline
// ray query; SceneAccess.hlsli has the other hooks) and writes the accumulation buffer, denoiser
// guides, debug views and, for ReSTIR, the primary hits and GI reservoirs.
// The ray tracing variants run TracePixel from a ray generation shader; their miss and closest-hit
// shaders only serve the SER variant's hit objects (the pipeline keeps them in every variant). With
// AIR_PT_COMPUTE (the cache's variants) a compute shader runs it instead: its rays are inline queries
// anyway, and drivers that run linear algebra in compute shaders may not in ray tracing pipelines.
#ifndef AIR_PT_SER
#define AIR_PT_SER 0
#endif
#ifndef AIR_PT_COMPUTE
#define AIR_PT_COMPUTE 0
#endif

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"
#include "ReSTIRShared.h"

[[vk::push_constant]] ConstantBuffer<TraceConstants> g : register(b0);

#if !AIR_PT_COMPUTE
// Payload access qualifiers tell the driver which stage reads and writes each field, so it keeps
// them live only where needed (always on from SM 6.7; the build enables them for 6.6 too).
struct [raypayload] Payload
{
    float t : read(caller) : write(caller, closesthit, miss); // < 0: miss
    uint instance : read(caller) : write(caller, closesthit);
    uint primitive : read(caller) : write(caller, closesthit);
    float2 barycentrics : read(caller) : write(caller, closesthit);
};
#endif

// --- Integrator hooks ----------------------------------------------------------------------------

#include "SceneAccess.hlsli"

// Closest hits come from an inline ray query: TraceRay into the closest-hit shader measured 19-29%
// slower on every test scene (it saves the megakernel's live state around each call), and twice as
// slow in the ReSTIR GI variant, whose state is larger. Only the SER variant traces into hit objects.
HitInfo TraceClosest(float3 origin, float3 direction, uint bounce)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;
    HitInfo hit;
#if AIR_PT_SER
    Payload payload;
    payload.t = -1.0f;
    payload.instance = 0;
    payload.primitive = 0;
    payload.barycentrics = float2(0.0f, 0.0f);
    // Trace into a hit object, regroup the threads by the instance they hit (misses together), then
    // read the hit straight from the object: no closest-hit invocation. Bounce rays scatter in all
    // directions, so neighbouring threads shade unrelated materials; after reordering a warp mostly
    // shades one, which keeps its loads and branches coherent.
    dx::HitObject hitObject = dx::HitObject::TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 1, 0, ray, payload);
    if (bounce > 0u) // primary rays are coherent already
    {
        const uint coherence = hitObject.IsHit() ? hitObject.GetInstanceID() + 1u : 0u;
        dx::MaybeReorderThread(hitObject, coherence, 16);
    }
    hit.hit = hitObject.IsHit();
    hit.t = hit.hit ? hitObject.GetRayTCurrent() : -1.0f;
    hit.instance = hitObject.GetInstanceID();
    hit.primitive = hitObject.GetPrimitiveIndex();
    BuiltInTriangleIntersectionAttributes attributes;
    hitObject.GetAttributes(attributes);
    hit.barycentrics = attributes.barycentrics;
#else
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    hit.hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    hit.t = hit.hit ? query.CommittedRayT() : -1.0f;
    hit.instance = hit.hit ? query.CommittedInstanceID() : 0u;
    hit.primitive = hit.hit ? query.CommittedPrimitiveIndex() : 0u;
    hit.barycentrics = hit.hit ? query.CommittedTriangleBarycentrics() : float2(0.0f, 0.0f);
#endif
    return hit;
}

#if AIR_PT_NRC
// --- Neural radiance cache hooks (NrcShared.h) ----------------------------------------------------

static NrcData gNrc;   // this frame's, set by TracePixel
static uint gNrcPath;  // the training path of the pixel's tile
static uint gNrcPixel; // y * width + x

float4 MlpLoadWeights4(uint index)
{
    ByteAddressBuffer parameters = ResourceDescriptorHeap[gNrc.network.parameters];
    return asfloat(parameters.Load4(index * 4u));
}

float2 MlpLoadFeatures(uint index)
{
    ByteAddressBuffer parameters = ResourceDescriptorHeap[gNrc.network.parameters];
    return asfloat(parameters.Load2(index * 4u));
}

#include "Nrc.hlsli"
#if AIR_PT_NRC == 1
#include "air/renderer/shader/MlpLinAlg.hlsli"
#endif

float3 NrcPredict(NrcQuery q)
{
    MlpVector x = NrcEncode(gNrc.grid, q);
#if AIR_PT_NRC == 1
    MlpLinAlgForward f = MlpLinAlgEvaluate(gNrc.network, MlpToHalf(x));
    float3 output = float3(f.output[0], f.output[1], f.output[2]);
#else
    MlpForward f = MlpEvaluate(x);
    float3 output = f.output.v[0].xyz;
#endif
    return NrcRadiance(q, output, gNrc.normalization);
}

NrcTrainingRecord NrcReadRecord(uint slot)
{
    RWStructuredBuffer<NrcTrainingRecord> records = ResourceDescriptorHeap[gNrc.trainingRecords];
    return records[slot];
}

void NrcWriteRecord(uint slot, NrcTrainingRecord record)
{
    RWStructuredBuffer<NrcTrainingRecord> records = ResourceDescriptorHeap[gNrc.trainingRecords];
    records[slot] = record;
}

void NrcDeferQuery(NrcQuery q, float3 throughput)
{
    RWStructuredBuffer<NrcPixelQuery> queries = ResourceDescriptorHeap[gNrc.pixelQueries];
    NrcPixelQuery p;
    p.query = q;
    p.throughput = throughput;
    p.pixel = gNrcPixel;
    queries[gNrcPath] = p;
}
#endif

#include "Integrator.hlsli"

// --- Debug helpers -------------------------------------------------------------------------------

// Every triangle the ray crosses, not just the closest: shows where geometry piles up.
uint CountIntersections(float3 origin, float3 direction)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    uint count = 0;
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            ++count; // never committed, so traversal visits every intersection
        }
    }
    return count;
}

float3 DebugValue(uint view, PathSample s, float intersections, FrameData f)
{
    switch (view)
    {
    case kDebugNormals: return s.hit ? s.normal * 0.5f + 0.5f : float3(0.0f, 0.0f, 0.0f);
    case kDebugBaseColor: return s.material.baseColor;
    case kDebugRoughness: return s.material.roughness;
    case kDebugMetallic: return s.material.metallic;
    case kDebugDepth: return s.hit ? saturate(s.hitDistance / f.debugScale) : 1.0f;
    case kDebugIntersections: return intersections / f.debugScale;
    case kDebugBounces: return float(s.bounces) / float(max(f.maxBounces, 1u));
    case kDebugDirect: return s.directRadiance;
    case kDebugIndirect: return s.radiance - s.directRadiance;
    default: return s.radiance;
    }
}

// --- Shaders -------------------------------------------------------------------------------------

void TracePixel(uint2 pixel)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    if (pixel.x >= f.width || pixel.y >= f.height)
    {
        return;
    }
    float2 size = float2(f.width, f.height);

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
    params.nrcSpread = f.nrc.spread;
    params.nrcSlot = 0u;
    params.nrcRecords = f.nrc.recordsPerPath;
    uint nrcFlags = 0u; // of the first sample: its pixel's training path, if it traces one
#if AIR_PT_NRC
    gNrc = f.nrc;
    if ((f.flags & kFlagNrc) != 0u)
    {
        uint2 tile = pixel / f.nrc.tile;
        gNrcPath = tile.y * f.nrc.tilesX + tile.x;
        gNrcPixel = pixel.y * f.width + pixel.x;
        if (tile.x < f.nrc.tilesX && tile.y < f.nrc.tilesY && pixel.x % f.nrc.tile == f.nrc.offsetX &&
            pixel.y % f.nrc.tile == f.nrc.offsetY)
        {
            params.nrcSlot = gNrcPath * f.nrc.recordsPerPath;
            bool unbiased = Pcg(gNrcPath ^ Pcg(f.nrc.frame)) % f.nrc.unbiasedPeriod == 0u;
            nrcFlags = kFlagNrcTrain | (unbiased ? kFlagNrcUnbiased : 0u);
            RWStructuredBuffer<NrcPixelQuery> queries = ResourceDescriptorHeap[f.nrc.pixelQueries];
            queries[gNrcPath].pixel = kNrcNoPixel; // until the path makes its render query
        }
    }
#endif

    uint seed = Pcg(pixel.x + Pcg(pixel.y + Pcg(f.frameIndex)));
    float3 sum = 0.0f;
    PathSample first = (PathSample)0;
    float2 firstPixel = float2(pixel) + 0.5f;
    for (uint i = 0; i < f.samplesPerPixel; ++i)
    {
        float2 jitter = (f.flags & kFlagGlobalJitter) != 0u ? f.jitter + 0.5f : Rand2(seed);
        float2 pixelPos = float2(pixel) + jitter;
        float3 direction = CameraRayDirection(f.camera, pixelPos, size);
        // ReSTIR GI and the cache's training take the first sample.
        params.flags = i == 0u ? f.flags | kFlagRecordGi | nrcFlags : f.flags;
        PathSample s = TracePath(f.camera.position.xyz, direction, params, seed);
        AddAerialPerspective(params, direction, s);
#if AIR_PT_NRC
        if ((params.flags & kFlagNrcTrain) != 0u && s.hit && params.sky.aerialDistance > 0.0f)
        {
            // The deferred render query's light crosses the same air as the rest of the sample.
            RWStructuredBuffer<NrcPixelQuery> queries = ResourceDescriptorHeap[f.nrc.pixelQueries];
            if (queries[gNrcPath].pixel != kNrcNoPixel)
            {
                queries[gNrcPath].throughput *= AerialPerspective(params.sky, direction, s.hitDistance).transmittance;
            }
        }
#endif
        float intersections = 0.0f;
        if (f.debugView == kDebugIntersections)
        {
            intersections = float(CountIntersections(f.camera.position.xyz, direction));
        }
        float3 value = f.debugView == kDebugNone ? s.radiance : DebugValue(f.debugView, s, intersections, f);
#if AIR_PT_NRC
        if (f.debugView == kDebugNrcCache)
        {
            // What the cache predicts the first surface reflects towards the camera (every thread
            // predicts, like TracePath's lookups).
            NrcQuery q;
            q.position = s.position;
            q.normal = NrcPackDirection(s.hit ? s.normal : float3(0.0f, 0.0f, 1.0f));
            q.direction = NrcPackDirection(-direction);
            q.roughness = NrcPackUnorm16x2(s.material.roughness, 0.0f);
            q.diffuse = NrcPackColor(s.diffuseAlbedo);
            q.specular = NrcPackColor(s.specularAlbedo);
            float3 prediction = NrcPredict(q);
            value = s.hit ? prediction : float3(0.0f, 0.0f, 0.0f);
        }
#endif
        sum += value;
        if (i == 0)
        {
            first = s;
            firstPixel = pixelPos;
        }
    }

    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
    float4 previous = (f.flags & kFlagReset) != 0u ? float4(0.0f, 0.0f, 0.0f, 0.0f) : accumulation[pixel];
    accumulation[pixel] = previous + float4(sum, float(f.samplesPerPixel));

    if ((f.flags & (kFlagReSTIR | kFlagReSTIRGI)) != 0u)
    {
        // Every sample left the direct light at its first surface out of `sum`; the ReSTIR passes
        // add the first sample's, times the sample count, so the average stays unbiased.
        RWStructuredBuffer<PrimaryHit> primaryHits = ResourceDescriptorHeap[g.primaryHits];
        PrimaryHit h;
        h.instance = first.instance;
        h.primitive = first.primitive;
        h.barycentrics = first.barycentrics;
        h.normalAndRoughness =
            EncodeNormalRoughness(first.hit ? first.normal : float3(0.0f, 0.0f, 1.0f), first.material.roughness);
        h.distance = first.hitDistance;
        primaryHits[pixel.y * f.width + pixel.x] = h;
    }
#if AIR_PT_RESTIR_GI
    if ((f.flags & kFlagReSTIRGI) != 0u)
    {
        // The first sample's indirect light as ReSTIR GI's initial reservoir (confidence 1).
        RWStructuredBuffer<GiReservoir> giReservoirs = ResourceDescriptorHeap[g.giReservoirs];
        GiReservoir r;
        r.instance = first.gi.valid ? first.gi.instance : kInvalidIndex;
        r.primitive = first.gi.primitive;
        r.barycentrics = first.gi.barycentrics;
        r.direction = first.gi.direction;
        r.lightPdf = first.gi.lightPdf;
        r.radianceMis = first.gi.radianceMis;
        r.weight = first.gi.valid ? first.gi.weight : 0.0f;
        r.radianceFixed = first.gi.radianceFixed;
        r.strategyAndConfidence = first.gi.strategy | (1u << 8);
        giReservoirs[pixel.y * f.width + pixel.x] = r;
    }
#endif

    if ((f.flags & kFlagGuides) != 0u)
    {
        RWTexture2D<float4> diffuseAlbedo = ResourceDescriptorHeap[g.diffuseAlbedo];
        RWTexture2D<float4> specularAlbedo = ResourceDescriptorHeap[g.specularAlbedo];
        RWTexture2D<float4> normalRoughness = ResourceDescriptorHeap[g.normalRoughness];
        RWTexture2D<float> viewDepth = ResourceDescriptorHeap[g.viewDepth];
        RWTexture2D<float2> motionVectors = ResourceDescriptorHeap[g.motionVectors];
        diffuseAlbedo[pixel] = float4(first.diffuseAlbedo, 1.0f);
        specularAlbedo[pixel] = float4(first.specularAlbedo, 1.0f);
        normalRoughness[pixel] = float4(first.normal, first.material.roughness);
        float3 direction = CameraRayDirection(f.camera, firstPixel, size);
        viewDepth[pixel] = first.hit ? first.hitDistance * dot(direction, f.camera.forward.xyz) : 65504.0f;
        // Static scene: motion comes from the camera only. Misses reproject their direction.
        float2 previousPixel = first.hit ? CameraProject(f.previousCamera, first.position, 1.0f, size)
                                         : CameraProject(f.previousCamera, direction, 0.0f, size);
        motionVectors[pixel] = previousPixel - firstPixel;
    }
}

#if AIR_PT_COMPUTE
[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    TracePixel(id.xy);
}
#else
[shader("raygeneration")]
void RayGen()
{
    TracePixel(DispatchRaysIndex().xy);
}

[shader("miss")]
void Miss(inout Payload payload)
{
    payload.t = -1.0f;
}

[shader("closesthit")]
void ClosestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    payload.t = RayTCurrent();
    payload.instance = InstanceID();
    payload.primitive = PrimitiveIndex();
    payload.barycentrics = attributes.barycentrics;
}
#endif
