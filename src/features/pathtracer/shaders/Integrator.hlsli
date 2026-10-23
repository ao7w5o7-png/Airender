// Unidirectional path tracer with next event estimation and multiple importance sampling.
// Shared by the GPU ray generation shader and the CPU reference renderer (ReferenceRenderer.cpp),
// so the two produce the same estimator and can be compared image to image.
//
// The including code defines, before including this file:
//   HitInfo       TraceClosest(float3 origin, float3 direction, uint bounce)
//   bool          TraceVisible(float3 origin, float3 direction, float tMax)
//   InstanceData  LoadInstance(uint instance)
//   uint          LoadInstanceLight(uint instance)   first light-list entry, kInvalidIndex if none
//   uint          LoadIndex(InstanceData instance, uint index)
//   Vertex        LoadVertex(InstanceData instance, uint index)
//   MaterialData  LoadMaterial(uint material)
//   LightTriangle LoadLight(uint light)
//   float         BsdfLutLoad(uint index)
// HitInfo, CameraData and the kFlag* constants come from PathTracerShared.h.
//
// Estimator: emitted radiance is gathered at path vertices 1 .. maxBounces + 1 (vertex 1 is the
// first hit). Every vertex below the limit samples one emissive triangle (power-proportional)
// and the BSDF; both strategies are weighted with the power heuristic. The environment is only
// reached by BSDF sampling. Russian roulette starts after the third bounce and is unbiased.
#ifndef AIR_PT_INTEGRATOR_HLSLI
#define AIR_PT_INTEGRATOR_HLSLI

#include "Bsdf.hlsli"
#include "Sampling.hlsli"

struct IntegratorParams
{
    float3 environment; // radiance, nits
    uint lightCount;
    uint maxBounces;
    uint flags; // kFlagFurnace, kFlagNoNee
};

struct Surface
{
    float3 position;
    float3 geometricNormal; // unit, facing the incoming ray
    float3 shadingNormal;   // unit, same side as geometricNormal
    float3 emission;        // radiance towards the incoming ray (0 when hit from the back)
    uint lightIndex;        // this triangle in the light list; kInvalidIndex when it does not emit
    BsdfMaterial material;
};

struct PathSample
{
    float3 radiance;
    float3 directRadiance; // part of `radiance` from paths with at most one bounce
    // First hit, for denoiser guides and debug views.
    bool hit;
    float hitDistance;
    float3 position;
    float3 normal;
    BsdfMaterial material;
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    uint bounces; // surfaces the path scattered at
};

float3 Xyz(float4 v)
{
    return float3(v.x, v.y, v.z);
}

float3 TransformRow3(float4 r0, float4 r1, float4 r2, float3 p, float w)
{
    float4 h = float4(p, w);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

// Offsets a ray origin off the surface along the normal of the side it leaves from.
float3 OffsetRayOrigin(float3 p, float3 n)
{
    float scale = 1e-4f * (1.0f + MaxComponent(abs(p)));
    return p + n * scale;
}

// --- Camera --------------------------------------------------------------------------------------

// Direction through pixel position `pixel` (0 .. size, y down).
float3 CameraRayDirection(CameraData c, float2 pixel, float2 size)
{
    float ndcX = pixel.x / size.x * 2.0f - 1.0f;
    float ndcY = 1.0f - pixel.y / size.y * 2.0f;
    float tanHalf = c.position.w;
    float aspect = c.forward.w;
    return normalize(Xyz(c.forward) + Xyz(c.right) * (ndcX * tanHalf * aspect) + Xyz(c.up) * (ndcY * tanHalf));
}

// Pixel position of a world point (or of a direction when w = 0), inverse of CameraRayDirection.
float2 CameraProject(CameraData c, float3 p, float w, float2 size)
{
    float3 d = p - Xyz(c.position) * w;
    float z = max(dot(d, Xyz(c.forward)), 1e-6f);
    float tanHalf = c.position.w;
    float aspect = c.forward.w;
    float ndcX = dot(d, Xyz(c.right)) / (z * tanHalf * aspect);
    float ndcY = dot(d, Xyz(c.up)) / (z * tanHalf);
    return float2((ndcX + 1.0f) * 0.5f * size.x, (1.0f - ndcY) * 0.5f * size.y);
}

// --- Lights --------------------------------------------------------------------------------------

// Chooses a light with its `probability` in O(1) from the alias table in the light list: one load
// when the drawn entry is kept, two when it defers to its alias. uAlias must be independent of
// uSelect (using the fraction of uSelect * count would quantize the choice for large lists).
LightTriangle SampleLight(float uSelect, float uAlias, uint count)
{
    uint i = min(uint(uSelect * float(count)), count - 1u);
    LightTriangle light = LoadLight(i);
    if (uAlias >= light.aliasThreshold)
    {
        light = LoadLight(light.alias);
    }
    return light;
}

// Solid angle pdf of NEE choosing the point at distance `dist` on `light`, seen under cosLight.
float LightPdf(LightTriangle light, float dist, float cosLight)
{
    return cosLight > 0.0f ? light.probability * dist * dist / (light.area * cosLight) : 0.0f;
}

// --- Surfaces ------------------------------------------------------------------------------------

Surface LoadSurface(HitInfo hit, float3 rayDirection, IntegratorParams params)
{
    InstanceData instance = LoadInstance(hit.instance);
    uint i0 = LoadIndex(instance, hit.primitive * 3u);
    uint i1 = LoadIndex(instance, hit.primitive * 3u + 1u);
    uint i2 = LoadIndex(instance, hit.primitive * 3u + 2u);
    Vertex v0 = LoadVertex(instance, i0);
    Vertex v1 = LoadVertex(instance, i1);
    Vertex v2 = LoadVertex(instance, i2);
    float b1 = hit.barycentrics.x;
    float b2 = hit.barycentrics.y;
    float b0 = 1.0f - b1 - b2;

    float3 p0 = TransformRow3(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, v0.position, 1.0f);
    float3 p1 = TransformRow3(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, v1.position, 1.0f);
    float3 p2 = TransformRow3(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, v2.position, 1.0f);
    float3 objectNormal = v0.normal * b0 + v1.normal * b1 + v2.normal * b2;

    Surface s;
    s.position = p0 * b0 + p1 * b1 + p2 * b2;
    float3 ng = normalize(cross(p1 - p0, p2 - p0)); // the emitting side of a light triangle
    bool frontFace = dot(ng, rayDirection) < 0.0f;
    s.geometricNormal = frontFace ? ng : -ng;
    float3 ns = normalize(TransformRow3(instance.normalToWorld0, instance.normalToWorld1, instance.normalToWorld2,
                                        objectNormal, 0.0f));
    s.shadingNormal = dot(ns, s.geometricNormal) < 0.0f ? -ns : ns;

    MaterialData m = LoadMaterial(instance.materialIndex);
    s.material.baseColor = m.baseColor;
    s.material.roughness = m.roughness;
    s.material.metallic = m.metallic;
    s.emission = frontFace ? m.emission : float3(0.0f, 0.0f, 0.0f);
    if ((params.flags & kFlagFurnace) != 0u)
    {
        s.material.baseColor = float3(1.0f, 1.0f, 1.0f);
        s.emission = float3(0.0f, 0.0f, 0.0f);
    }
    s.lightIndex = any(s.emission) ? LoadInstanceLight(hit.instance) + hit.primitive : kInvalidIndex;
    return s;
}

// --- Path ----------------------------------------------------------------------------------------

PathSample TracePath(float3 origin, float3 direction, IntegratorParams params, AIR_INOUT(uint) seed)
{
    PathSample result;
    result.radiance = float3(0.0f, 0.0f, 0.0f);
    result.directRadiance = float3(0.0f, 0.0f, 0.0f);
    result.hit = false;
    result.hitDistance = 0.0f;
    result.position = float3(0.0f, 0.0f, 0.0f);
    result.normal = float3(0.0f, 0.0f, 0.0f);
    result.material.baseColor = float3(0.0f, 0.0f, 0.0f);
    result.material.roughness = 0.0f;
    result.material.metallic = 0.0f;
    result.diffuseAlbedo = float3(0.0f, 0.0f, 0.0f);
    result.specularAlbedo = float3(0.0f, 0.0f, 0.0f);
    result.bounces = 0u;

    bool useNee = params.lightCount > 0u && (params.flags & kFlagNoNee) == 0u && (params.flags & kFlagFurnace) == 0u;
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float previousBsdfPdf = 0.0f; // solid angle pdf of the BSDF sample that produced `direction`

    for (uint bounce = 0u; bounce <= params.maxBounces; ++bounce)
    {
        HitInfo hit = TraceClosest(origin, direction, bounce);
        bool direct = bounce <= 1u;
        if (!hit.hit)
        {
            float3 environment = throughput * params.environment;
            result.radiance += environment;
            if (direct)
            {
                result.directRadiance += environment;
            }
            break;
        }
        Surface s = LoadSurface(hit, direction, params);
        if (bounce == 0u)
        {
            result.hit = true;
            result.hitDistance = hit.t;
            result.position = s.position;
            result.normal = s.shadingNormal;
            result.material = s.material;
            result.diffuseAlbedo = DiffuseAlbedo(s.material);
            result.specularAlbedo = SpecularAlbedo(s.material, saturate(dot(s.shadingNormal, -direction)));
        }

        // Emission found by the BSDF sample, weighted against NEE having found it.
        if (any(s.emission))
        {
            float weight = 1.0f;
            if (bounce > 0u && useNee)
            {
                LightTriangle light = LoadLight(s.lightIndex);
                float lightPdf = LightPdf(light, hit.t, dot(-direction, s.geometricNormal));
                weight = PowerHeuristic(previousBsdfPdf, lightPdf);
            }
            float3 emitted = throughput * s.emission * weight;
            result.radiance += emitted;
            if (direct)
            {
                result.directRadiance += emitted;
            }
        }
        if (bounce == params.maxBounces)
        {
            break;
        }
        result.bounces = bounce + 1u;

        float3 ns = s.shadingNormal;
        float3 ng = s.geometricNormal;
        float3 wo = -direction;
        if (dot(ns, wo) <= 0.0f)
        {
            ns = ng; // the interpolated normal faces away from the viewer: fall back to the facet
        }
        Frame frame = MakeFrame(ns);
        float3 woLocal = ToLocal(frame, wo);
        float3 rayOrigin = OffsetRayOrigin(s.position, ng);

        // Next event estimation: one point on one emissive triangle.
        if (useNee)
        {
            float uLight = Rand(seed);
            float uAlias = Rand(seed);
            float2 uPoint = float2(Rand(seed), Rand(seed));
            LightTriangle light = SampleLight(uLight, uAlias, params.lightCount);
            float2 b = SampleTriangle(uPoint);
            float3 lightPoint = light.p0 + light.e1 * b.x + light.e2 * b.y;
            float3 toLight = lightPoint - rayOrigin;
            float dist = length(toLight);
            float3 wi = toLight / dist;
            float3 lightNormal = normalize(cross(light.e1, light.e2));
            float cosLight = dot(-wi, lightNormal);
            if (cosLight > 0.0f && dot(wi, ng) > 0.0f)
            {
                BsdfEval e = EvaluateBsdf(s.material, woLocal, ToLocal(frame, wi));
                float lightPdf = LightPdf(light, dist, cosLight);
                if (any(e.fCos) && lightPdf > 0.0f && TraceVisible(rayOrigin, wi, dist * (1.0f - 1e-4f)))
                {
                    float weight = PowerHeuristic(lightPdf, e.pdf);
                    float3 contribution = throughput * e.fCos * light.radiance * (weight / lightPdf);
                    result.radiance += contribution;
                    if (bounce == 0u)
                    {
                        result.directRadiance += contribution;
                    }
                }
            }
        }

        // Continue the path by sampling the BSDF.
        float uLobe = Rand(seed);
        float2 u = float2(Rand(seed), Rand(seed));
        BsdfSample bs = SampleBsdf(s.material, woLocal, uLobe, u);
        if (!bs.valid)
        {
            break;
        }
        float3 wiWorld = ToWorld(frame, bs.wi);
        if (dot(wiWorld, ng) <= 0.0f)
        {
            break; // shading normal sent the ray into the surface
        }
        throughput *= bs.weight;
        previousBsdfPdf = bs.pdf;
        origin = rayOrigin;
        direction = wiWorld;

        if (bounce >= 3u)
        {
            float survive = min(MaxComponent(throughput), 0.95f);
            if (Rand(seed) >= survive)
            {
                break;
            }
            throughput /= survive;
        }
    }
    return result;
}

#endif
