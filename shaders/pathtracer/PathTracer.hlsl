// GPU path tracer, hardware ray tracing (DXR 1.1 / VK_KHR_ray_tracing_pipeline).
// Compiled by DXC as a library:  dxc -T lib_6_6 PathTracer.hlsl            -> DXIL
//                                dxc -T lib_6_6 -spirv -fspv-target-env=vulkan1.3 ... -> SPIR-V
//
// Design: the bounce loop lives in raygen and TraceRay is never recursive (maxRecursionDepth = 1),
// which keeps the stack small and lets the driver schedule the whole path in one shader.
// Resources are bindless (SM 6.6 ResourceDescriptorHeap), indices come from push constants.

#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<TraceConstants> g : register(b0);

static const float kPi = 3.14159265358979f;

struct Payload {
    float3 hitPosition;
    float hitT;          // < 0: miss
    float3 normal;       // world space, facing the ray
    uint materialIndex;
};

// ---------------------------------------------------------------------------
// RNG: PCG hash, decorrelated per pixel / frame / sample
// ---------------------------------------------------------------------------
uint Pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float Rand(inout uint seed) {
    seed = Pcg(seed);
    return float(seed) * (1.0f / 4294967296.0f);
}

float3x3 TangentFrame(float3 n) {
    // Duff et al. 2017, "Building an Orthonormal Basis, Revisited"
    float s = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float b = n.x * n.y * a;
    float3 t = float3(1.0f + s * n.x * n.x * a, s * b, -s * n.x);
    float3 bt = float3(b, s + n.y * n.y * a, -n.y);
    return float3x3(t, bt, n);
}

float3 SampleCosineHemisphere(float3 n, float2 u) {
    float r = sqrt(u.x);
    float phi = 2.0f * kPi * u.y;
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - u.x)));
    return mul(local, TangentFrame(n));
}

float3 Sky(float3 dir) {
    float t = saturate(dir.y * 0.5f + 0.5f);
    return lerp(float3(1.0f, 1.0f, 1.0f), float3(0.5f, 0.7f, 1.0f), t);
}

[shader("raygeneration")]
void RayGen() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 size = DispatchRaysDimensions().xy;
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[g.materialBuffer];
    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulationUav];

    uint seed = Pcg(pixel.x + Pcg(pixel.y + Pcg(g.frameIndex)));
    float3 radianceSum = 0.0f;

    for (uint s = 0; s < g.samplesPerPixel; ++s) {
        // Pinhole camera with per-sample sub-pixel jitter (box filter AA).
        float2 jitter = float2(Rand(seed), Rand(seed));
        float2 ndc = ((float2(pixel) + jitter) / float2(size)) * 2.0f - 1.0f;
        float tanHalfFov = tan(g.cameraPositionFov.w * 0.5f);
        float aspect = float(size.x) / float(size.y);
        float3 dir = normalize(g.cameraForward.xyz + ndc.x * tanHalfFov * aspect * g.cameraRight.xyz -
                               ndc.y * tanHalfFov * g.cameraUp.xyz);

        RayDesc ray;
        ray.Origin = g.cameraPositionFov.xyz;
        ray.Direction = dir;
        ray.TMin = 0.0f;
        ray.TMax = 1e30f;

        float3 throughput = 1.0f;
        float3 radiance = 0.0f;
        for (uint bounce = 0; bounce <= g.maxBounces; ++bounce) {
            Payload payload = (Payload)0;
            TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

            if (payload.hitT < 0.0f) {
                radiance += throughput * Sky(ray.Direction);
                break;
            }
            MaterialData material = materials[payload.materialIndex];
            radiance += throughput * material.emission;

            // Lambertian: cosine sampling cancels cos/pdf, throughput *= albedo.
            // TODO: GGX specular, next event estimation + MIS, textures.
            throughput *= material.baseColor;

            // Russian roulette after a few bounces keeps long paths unbiased and cheap.
            if (bounce >= 3) {
                float p = min(max(throughput.x, max(throughput.y, throughput.z)), 0.95f);
                if (Rand(seed) > p) {
                    break;
                }
                throughput /= p;
            }

            ray.Origin = payload.hitPosition + payload.normal * 1e-4f;
            ray.Direction = SampleCosineHemisphere(payload.normal, float2(Rand(seed), Rand(seed)));
            ray.TMin = 0.0f;
        }
        radianceSum += radiance;
    }

    float4 previous = g.resetAccumulation != 0 ? float4(0, 0, 0, 0) : accumulation[pixel];
    accumulation[pixel] = previous + float4(radianceSum, float(g.samplesPerPixel));
}

[shader("miss")]
void Miss(inout Payload payload) { payload.hitT = -1.0f; }

[shader("closesthit")]
void ClosestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    StructuredBuffer<InstanceData> instances = ResourceDescriptorHeap[g.instanceBuffer];
    InstanceData instance = instances[InstanceID()];
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[instance.vertexBuffer];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[instance.indexBuffer];

    uint base = PrimitiveIndex() * 3;
    float3 n0 = vertices[indices[base + 0]].normal;
    float3 n1 = vertices[indices[base + 1]].normal;
    float3 n2 = vertices[indices[base + 2]].normal;
    float3 bary = float3(1.0f - attributes.barycentrics.x - attributes.barycentrics.y, attributes.barycentrics.x,
                         attributes.barycentrics.y);
    float3 objectNormal = n0 * bary.x + n1 * bary.y + n2 * bary.z;
    float3 worldNormal = normalize(mul(objectNormal, (float3x3)WorldToObject3x4()));

    payload.hitT = RayTCurrent();
    payload.hitPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.normal = dot(worldNormal, WorldRayDirection()) < 0.0f ? worldNormal : -worldNormal;
    payload.materialIndex = instance.materialIndex;
}
