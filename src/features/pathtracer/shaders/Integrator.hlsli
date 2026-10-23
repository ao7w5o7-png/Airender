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
//   LightTreeNode LoadLightNode(uint node)
//   float         BsdfLutLoad(uint index)
// HitInfo, CameraData and the kFlag* constants come from PathTracerShared.h.
//
// Estimator: emitted radiance is gathered at path vertices 1 .. maxBounces + 1 (vertex 1 is the
// first hit). Every vertex below the limit samples one emissive triangle and the BSDF; both
// strategies are weighted with the power heuristic. The first lightTreeBounces vertices choose the
// triangle with the light tree (power, distance and orientation as seen from the vertex: far fewer
// wasted samples among many lights, but a walk of ~log2(lights) dependent loads); deeper vertices,
// whose light reaches the image after a bounce, choose by power alone in O(1). The environment is
// only reached by BSDF sampling. Russian roulette starts after the third bounce and is unbiased.
// With kFlagReSTIR the direct light from emissive triangles at the first surface is left out
// entirely: ReSTIR DI (ReSTIR.hlsli) estimates it.
#ifndef AIR_PT_INTEGRATOR_HLSLI
#define AIR_PT_INTEGRATOR_HLSLI

#include "Bsdf.hlsli"
#include "Sampling.hlsli"

struct IntegratorParams
{
    float3 environment; // radiance, nits
    uint lightCount;
    uint maxBounces;
    uint flags;            // kFlagFurnace, kFlagNoNee
    uint lightTreeBounces; // vertices 0 .. lightTreeBounces - 1 choose lights with the light tree
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
    // The first hit as TraceClosest reported it (instance kInvalidIndex: none), for ReSTIR DI.
    uint instance;
    uint primitive;
    float2 barycentrics;
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

// The light-list index SampleLight(uSelect, uAlias, count) chooses.
uint SampleLightIndex(float uSelect, float uAlias, uint count)
{
    uint i = min(uint(uSelect * float(count)), count - 1u);
    LightTriangle light = LoadLight(i);
    return uAlias >= light.aliasThreshold ? light.alias : i;
}

// Solid angle pdf of NEE choosing, with probability `pmf`, a light of `area` and the point on it at
// distance `dist`, seen under cosLight.
float LightPdf(float pmf, float area, float dist, float cosLight)
{
    return cosLight > 0.0f ? pmf * dist * dist / (area * cosLight) : 0.0f;
}

// --- Light tree ----------------------------------------------------------------------------------

float SafeSqrt(float v)
{
    return sqrt(max(v, 0.0f));
}

// cos(a - b) with the angle difference clamped at 0, from the sines and cosines of a and b.
float CosSubClamped(float sinA, float cosA, float sinB, float cosB)
{
    return cosA > cosB ? 1.0f : cosA * cosB + sinA * sinB;
}

float SinSubClamped(float sinA, float cosA, float sinB, float cosB)
{
    return cosA > cosB ? 0.0f : sinA * cosB - cosA * sinB;
}

// Importance of the emitters below `node` for point p with normal n (n = 0: no receiver cosine):
// power x a bound of the emitter cosine towards p x a bound of the receiver cosine / distance^2.
// The angles widen by the cone under which p sees the node's box; the distance is clamped so the
// importance stays finite inside the box. Zero only when no emitter below can light p: none faces
// it, or (the BSDFs only reflect) all are below its horizon.
float LightNodeImportance(LightTreeNode node, float3 p, float3 n)
{
    float3 center = (node.boundsMin + node.boundsMax) * 0.5f;
    float3 diagonal = node.boundsMax - node.boundsMin;
    float3 toPoint = p - center;
    // Emitters sharing one normal (a row of ceiling panels): exact test of p against the side of the
    // box they emit from, which the bounding sphere below would blur for a long, flat group.
    if (node.cosThetaO >= 0.9999f && dot(node.axis, toPoint) + dot(abs(node.axis), diagonal * 0.5f) <= 0.0f)
    {
        return 0.0f;
    }
    float distance2 = dot(toPoint, toPoint);
    float3 wi = toPoint * rsqrt(max(distance2, 1e-30f));
    float cosThetaW = dot(node.axis, wi);
    float sinThetaW = SafeSqrt(1.0f - cosThetaW * cosThetaW);
    // Directions from p into the box's bounding sphere: all of them from inside it.
    float radius2 = dot(diagonal, diagonal) * 0.25f;
    float cosThetaB = distance2 <= radius2 ? -1.0f : SafeSqrt(1.0f - radius2 / distance2);
    float sinThetaB = SafeSqrt(1.0f - cosThetaB * cosThetaB);
    float sinThetaO = SafeSqrt(1.0f - node.cosThetaO * node.cosThetaO);
    float cosThetaX = CosSubClamped(sinThetaW, cosThetaW, sinThetaO, node.cosThetaO);
    float sinThetaX = SinSubClamped(sinThetaW, cosThetaW, sinThetaO, node.cosThetaO);
    float cosThetaP = CosSubClamped(sinThetaX, cosThetaX, sinThetaB, cosThetaB);
    if (cosThetaP <= node.cosThetaE)
    {
        return 0.0f;
    }
    float importance = node.power * cosThetaP / max(distance2, length(diagonal) * 0.5f);
    if (any(n))
    {
        // Exact horizon test on the box (its support along n): a long, flat group of lights below
        // the tangent plane has a bounding sphere reaching far above it, which would keep it.
        if (dot(n, center - p) + dot(abs(n), diagonal * 0.5f) <= 0.0f)
        {
            return 0.0f;
        }
        float cosThetaI = dot(-wi, n); // towards the node
        float sinThetaI = SafeSqrt(1.0f - cosThetaI * cosThetaI);
        importance *= max(CosSubClamped(sinThetaI, cosThetaI, sinThetaB, cosThetaB), 0.0f);
    }
    return max(importance, 0.0f);
}

struct LightChoice
{
    uint light; // kInvalidIndex: no light can reach the point
    float pmf;
};

// Walks the light tree from the root, choosing each child by its importance at p (one uniform,
// rescaled at every level).
LightChoice SampleLightTree(float3 p, float3 n, float u)
{
    LightChoice choice;
    choice.light = kInvalidIndex;
    choice.pmf = 0.0f;
    uint index = 0u;
    LightTreeNode node = LoadLightNode(0u);
    float pmf = 1.0f;
    for (uint depth = 0u; depth <= 64u; ++depth)
    {
        if (node.isLeaf != 0u)
        {
            if (index > 0u || LightNodeImportance(node, p, n) > 0.0f)
            {
                choice.light = node.child;
                choice.pmf = pmf;
            }
            return choice;
        }
        LightTreeNode first = LoadLightNode(index + 1u);
        LightTreeNode second = LoadLightNode(node.child);
        float i0 = LightNodeImportance(first, p, n);
        float i1 = LightNodeImportance(second, p, n);
        if (i0 + i1 <= 0.0f)
        {
            return choice;
        }
        float p0 = i0 / (i0 + i1);
        if (u < p0)
        {
            u = min(u / p0, 0.99999994f);
            pmf *= p0;
            index = index + 1u;
            node = first;
        }
        else
        {
            u = min((u - p0) / (1.0f - p0), 0.99999994f);
            pmf *= 1.0f - p0;
            index = node.child;
            node = second;
        }
    }
    return choice;
}

// The probability SampleLightTree(p, n, u) chooses `light`: follows its trail from the root.
float LightTreePmf(uint light, float3 p, float3 n)
{
    LightTriangle l = LoadLight(light);
    if (l.treeDepth == kInvalidIndex)
    {
        return 0.0f;
    }
    uint index = 0u;
    LightTreeNode node = LoadLightNode(0u);
    float pmf = 1.0f;
    for (uint depth = 0u; depth < l.treeDepth; ++depth)
    {
        LightTreeNode first = LoadLightNode(index + 1u);
        LightTreeNode second = LoadLightNode(node.child);
        float i0 = LightNodeImportance(first, p, n);
        float i1 = LightNodeImportance(second, p, n);
        if (i0 + i1 <= 0.0f)
        {
            return 0.0f;
        }
        uint bit = depth < 32u ? (l.treeTrailLow >> depth) & 1u : (l.treeTrailHigh >> (depth - 32u)) & 1u;
        pmf *= (bit != 0u ? i1 : i0) / (i0 + i1);
        if (bit != 0u) // (HLSL has no conditional operator on structs)
        {
            index = node.child;
            node = second;
        }
        else
        {
            index = index + 1u;
            node = first;
        }
    }
    if (l.treeDepth == 0u && LightNodeImportance(node, p, n) <= 0.0f)
    {
        return 0.0f; // the only light, facing away (sampling finds nothing either)
    }
    return pmf;
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
    result.instance = kInvalidIndex;
    result.primitive = 0u;
    result.barycentrics = float2(0.0f, 0.0f);

    bool useNee = params.lightCount > 0u && (params.flags & kFlagNoNee) == 0u && (params.flags & kFlagFurnace) == 0u;
    // ReSTIR DI estimates the whole direct light at the first surface (from emissive triangles): the
    // path neither samples a light there nor counts the emission its BSDF sample reaches.
    bool restir = useNee && (params.flags & kFlagReSTIR) != 0u;
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float previousBsdfPdf = 0.0f; // solid angle pdf of the BSDF sample that produced `direction`
    // The vertex that produced `direction` and how NEE chose lights there (for MIS).
    float3 previousPosition = origin;
    float3 previousNormal = float3(0.0f, 0.0f, 0.0f);
    bool previousTree = false;

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
            result.instance = hit.instance;
            result.primitive = hit.primitive;
            result.barycentrics = hit.barycentrics;
        }

        // Emission found by the BSDF sample, weighted against NEE having found it.
        if (any(s.emission) && !(restir && bounce == 1u))
        {
            float weight = 1.0f;
            if (bounce > 0u && useNee)
            {
                LightTriangle light = LoadLight(s.lightIndex);
                float pmf = previousTree ? LightTreePmf(s.lightIndex, previousPosition, previousNormal) : light.probability;
                float lightPdf = LightPdf(pmf, light.area, hit.t, dot(-direction, s.geometricNormal));
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
        // The BSDF is prepared per call below: keeping a BsdfContext alive across the shadow ray
        // costs registers, which measured slower than repeating its cached LUT lookups.
        float3 woLocal = ToLocal(frame, wo);
        float3 rayOrigin = OffsetRayOrigin(s.position, ng);
        bool lightTree = bounce < params.lightTreeBounces;

        // Next event estimation: one point on one emissive triangle.
        if (useNee && !(restir && bounce == 0u))
        {
            float uLight = Rand(seed);
            float uAlias = Rand(seed); // drawn in both modes: the random sequence stays the same
            float2 uPoint = float2(Rand(seed), Rand(seed));
            LightTriangle light = LoadLight(0u); // the list has an entry even without lights
            float pmf = 0.0f;
            if (lightTree)
            {
                LightChoice choice = SampleLightTree(s.position, ns, uLight);
                if (choice.light != kInvalidIndex)
                {
                    light = LoadLight(choice.light);
                    pmf = choice.pmf;
                }
            }
            else
            {
                light = SampleLight(uLight, uAlias, params.lightCount);
                pmf = light.probability;
            }
            float2 b = SampleTriangle(uPoint);
            float3 lightPoint = light.p0 + light.e1 * b.x + light.e2 * b.y;
            float3 toLight = lightPoint - rayOrigin;
            float dist = length(toLight);
            float3 wi = toLight / dist;
            float3 lightNormal = normalize(cross(light.e1, light.e2));
            float cosLight = dot(-wi, lightNormal);
            if (pmf > 0.0f && cosLight > 0.0f && dot(wi, ng) > 0.0f)
            {
                BsdfEval e = EvaluateBsdf(s.material, woLocal, ToLocal(frame, wi));
                float lightPdf = LightPdf(pmf, light.area, dist, cosLight);
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
        previousPosition = s.position;
        previousNormal = ns;
        previousTree = lightTree;
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
