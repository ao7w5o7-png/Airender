// GPU path tracer: DXR 1.1 ray generation + closest hit, bindless resources (SM 6.6).
// The estimator itself is Integrator.hlsli, shared with the CPU reference renderer. This file
// supplies the platform hooks (TraceRay for closest hits, inline RayQuery for shadow rays) and
// writes the accumulation buffer, denoiser guides and debug views.

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<TraceConstants> g : register(b0);

struct Payload {
    float t; // < 0: miss
    uint instance;
    uint primitive;
    float2 barycentrics;
};

// --- Integrator hooks ----------------------------------------------------------------------------

float BsdfLutLoad(uint index) {
    StructuredBuffer<float> lut = ResourceDescriptorHeap[g.bsdfLut];
    return lut[index];
}

InstanceData LoadInstance(uint instance) {
    StructuredBuffer<InstanceData> instances = ResourceDescriptorHeap[g.instances];
    return instances[instance];
}

uint LoadIndex(InstanceData instance, uint index) {
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(instance.indexBuffer)];
    return indices[index];
}

Vertex LoadVertex(InstanceData instance, uint index) {
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(instance.vertexBuffer)];
    return vertices[index];
}

MaterialData LoadMaterial(uint material) {
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[g.materials];
    return materials[material];
}

LightTriangle LoadLight(uint light) {
    StructuredBuffer<LightTriangle> lights = ResourceDescriptorHeap[g.lights];
    return lights[light];
}

HitInfo TraceClosest(float3 origin, float3 direction) {
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;
    Payload payload;
    payload.t = -1.0f;
    payload.instance = 0;
    payload.primitive = 0;
    payload.barycentrics = float2(0.0f, 0.0f);
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 1, 0, ray, payload);
    HitInfo hit;
    hit.hit = payload.t >= 0.0f;
    hit.t = payload.t;
    hit.instance = payload.instance;
    hit.primitive = payload.primitive;
    hit.barycentrics = payload.barycentrics;
    return hit;
}

bool TraceVisible(float3 origin, float3 direction, float tMax) {
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    return query.CommittedStatus() == COMMITTED_NOTHING;
}

#include "Integrator.hlsli"

// --- Debug helpers -------------------------------------------------------------------------------

// Every triangle the ray crosses, not just the closest: shows where geometry piles up.
uint CountIntersections(float3 origin, float3 direction) {
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    uint count = 0;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            ++count; // never committed, so traversal visits every intersection
        }
    }
    return count;
}

float3 DebugValue(uint view, PathSample s, float intersections, FrameData f) {
    switch (view) {
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

[shader("raygeneration")]
void RayGen() {
    uint2 pixel = DispatchRaysIndex().xy;
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    float2 size = float2(f.width, f.height);

    IntegratorParams params;
    params.environment = f.environment.rgb;
    params.lightCount = f.lightCount;
    params.maxBounces = f.maxBounces;
    params.flags = f.flags;

    uint seed = Pcg(pixel.x + Pcg(pixel.y + Pcg(f.frameIndex)));
    float3 sum = 0.0f;
    PathSample first = (PathSample)0;
    float2 firstPixel = float2(pixel) + 0.5f;
    for (uint i = 0; i < f.samplesPerPixel; ++i) {
        float2 jitter = (f.flags & kFlagGlobalJitter) != 0u ? f.jitter + 0.5f : float2(Rand(seed), Rand(seed));
        float2 pixelPos = float2(pixel) + jitter;
        float3 direction = CameraRayDirection(f.camera, pixelPos, size);
        PathSample s = TracePath(f.camera.position.xyz, direction, params, seed);
        float intersections = 0.0f;
        if (f.debugView == kDebugIntersections) {
            intersections = float(CountIntersections(f.camera.position.xyz, direction));
        }
        sum += f.debugView == kDebugNone ? s.radiance : DebugValue(f.debugView, s, intersections, f);
        if (i == 0) {
            first = s;
            firstPixel = pixelPos;
        }
    }

    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
    float4 previous = (f.flags & kFlagReset) != 0u ? float4(0.0f, 0.0f, 0.0f, 0.0f) : accumulation[pixel];
    accumulation[pixel] = previous + float4(sum, float(f.samplesPerPixel));

    if ((f.flags & kFlagGuides) != 0u) {
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

[shader("miss")]
void Miss(inout Payload payload) {
    payload.t = -1.0f;
}

[shader("closesthit")]
void ClosestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    payload.t = RayTCurrent();
    payload.instance = InstanceID();
    payload.primitive = PrimitiveIndex();
    payload.barycentrics = attributes.barycentrics;
}
