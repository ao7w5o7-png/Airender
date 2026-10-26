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
//   TextureSetData LoadTextureSet(uint table, uint set)  a material's textures (IntegratorParams::textureSets)
//   float4        MaterialTextureSample(uint texture, float2 uv)  one of them, bilinear, repeating
// and, compiled with AIR_PT_NTC (the GPU's, where LibNTC's shaders were found):
//   void          NeuralTextureSample(TextureSetData t, float2 uv, float2 u, out float4 color, out float4 surface)
//                                                     a neural set's (TextureSetData::latents) texels, the
//                                                     texel stochastic filtering picks with u
//   LightTriangle LoadLight(uint light)
//   LightTreeNode LoadLightNode(uint node)
//   float         BsdfLutLoad(uint index)
//   float3        SkyTableLoad(uint texel)            the sky radiance table (SkyLookup.hlsli)
//   float4        AerialInscatterSample(SkyData sky, float3 uvw), AerialTransmittanceSample(...)
//                                                     the aerial perspective tables (SkyLookup.hlsli)
//   float         CloudBaseSample(CloudData c, float3 uvw), CloudDetailSample(...),
//   float         CloudMajorantLoad(CloudData c, uint cell)   the clouds (CloudLookup.hlsli)
//   VolumeData    LoadVolume(uint table, uint index)  the scene's volumes (IntegratorParams::volumeTable)
//   float         VolumeDensitySample(VolumeData v, float3 uvw), VolumeMajorantLoad(VolumeData v, uint cell)
//                                                     a volume's voxels (VolumeLookup.hlsli)
// and, compiled with AIR_PT_VOLUME_SHADOWS (the real-time volumes' kernels, VolumeMarchShared.h):
//   float         VolumeSunShadow(float3 p), VolumeGroupShadow(uint group, float3 p, float3 lightPoint)
//                                                     what the volumes let through of the sun's / a light
//                                                     group's light at p (their moment shadow maps; from
//                                                     lightPoint on the group's emitter)
// and, compiled with AIR_PT_PROBE (their probes' kernel):
//   bool          VolumeExplicitGroup(uint group)     the ray march lights the volumes with this group
// and, compiled with AIR_PT_NRC (the neural radiance cache's kernels, NrcShared.h):
//   float3        NrcPredict(NrcQuery q)              the radiance the cache predicts a vertex reflects
//   NrcTrainingRecord NrcReadRecord(uint slot), void NrcWriteRecord(uint slot, NrcTrainingRecord r)
//   void          NrcDeferQuery(NrcQuery q, float3 throughput)   a training path's render query
// HitInfo, CameraData and the kFlag* constants come from PathTracerShared.h.
//
// Estimator: emitted radiance is gathered at path vertices 1 .. maxBounces + 1 (vertex 1 is the
// first hit). Every vertex below the limit samples one emissive triangle and the BSDF; both
// strategies are weighted with the power heuristic. The first lightTreeBounces vertices choose the
// triangle with the light tree (power, distance and orientation as seen from the vertex: far fewer
// wasted samples among many lights, but a walk of ~log2(lights) dependent loads); deeper vertices,
// whose light reaches the image after a bounce, choose by power alone in O(1). The sky (a table, or
// uniform) is only reached by BSDF sampling; the sun's disc is a light of its own, sampled at every
// vertex and weighed against BSDF samples that reach it, like the triangles. Russian roulette
// starts after the third bounce and is unbiased.
// Rays that leave the geometry cross the clouds (CloudLookup.hlsli) on their way to the sky: a random
// walk through them gathers the light they scatter (EscapeRadiance), and the sun's light reaching a
// surface is dimmed by their transmittance. The clouds lie above all geometry, so the only other
// segment that can meet them is a camera ray from above their base.
// Volumes among the surfaces (VolumeLookup.hlsli: smoke, clouds from files) are crossed by every
// ray: delta tracking may stop a ray inside one, where the path scatters (Henyey-Greenstein) with
// next event estimation of the lights and the sun weighed against phase sampling, like at a surface;
// NEE from surfaces and from inside them multiplies the light by their transmittance (ratio
// tracking). Volumes may overlap: each is tracked on its own, the nearest collision wins (the first
// collision in a sum of media is the nearest of each medium's first collision, and the medium it is
// in is the one that scattered), and transmittances multiply. Scattering events are path vertices
// too, but they do not count as bounces: a path takes up to volumeEvents of them on top of its
// maxBounces surfaces (a walk through a bright medium needs hundreds), and Russian roulette only ends
// a walk once its throughput is low, as in the clouds. Compiled only with AIR_PT_VOLUME
// (PathTracerVolume.hlsl, the CPU reference); ReSTIR and the radiance cache stay off while the scene
// has volumes (PathTracer.cpp).
// The air (aerial perspective, SkyLookup.hlsli) dims and adds to what a camera ray sees on its
// surface (AddAerialPerspective, applied by the caller) and what any ray sees on the clouds; the sky
// table includes it already. Segments between surfaces of the scene, metres to hundreds of metres
// long, are taken as clear.
// With kFlagReSTIR the direct light from emissive triangles at the first surface is left out
// entirely: ReSTIR DI (ReSTIRDI.hlsli) estimates it.
// With kFlagNrc (AIR_PT_NRC kernels) a path ends at the first vertex where its footprint has spread
// past nrcSpread times the primary vertex's (Mueller et al. 2021), or at its last vertex: the light that
// vertex reflects comes from the neural radiance cache, predicted once the path is done (every thread
// of a wave then queries together). A training path (kFlagNrcTrain) defers its render query, goes on
// for a suffix that ends the same way (self-training; never with kFlagNrcUnbiased), and records every
// vertex as a training sample of what it reflected.
#ifndef AIR_PT_INTEGRATOR_HLSLI
#define AIR_PT_INTEGRATOR_HLSLI

#include "Bsdf.hlsli"
#include "Camera.hlsli"
#include "Sampling.hlsli"
#include "air/renderer/shader/CloudLookup.hlsli"
#include "air/renderer/shader/SkyLookup.hlsli"
#include "air/renderer/shader/VolumeLookup.hlsli"

// ReSTIR GI's candidate recording (kFlagReSTIRGI) is compiled only where it is used (PathTracerGi.hlsl):
// its state halves the speed of the plain kernel through register pressure.
#ifndef AIR_PT_RESTIR_GI
#define AIR_PT_RESTIR_GI 0
#endif
// The neural radiance cache: 1 with linear algebra (PathTracerNrc.hlsl), 2 in float32
// (PathTracerNrcPortable.hlsl), 0 compiled out.
#ifndef AIR_PT_NRC
#define AIR_PT_NRC 0
#endif
// The scene's volume likewise (PathTracerVolume.hlsl): tracking it, even past a volume that is not
// there, costs the plain kernel about 5%.
#ifndef AIR_PT_VOLUME
#define AIR_PT_VOLUME 0
#endif
// The real-time volumes (PathTracerVolumeShadows.hlsl, docs/architecture.md §8.13): the volumes are
// not tracked, a ray march renders them over the surfaces (VolumeMarch.hlsli), and the light reaching
// a surface through them comes from their moment shadow maps.
#ifndef AIR_PT_VOLUME_SHADOWS
#define AIR_PT_VOLUME_SHADOWS 0
#endif
// Their probes (VolumeProbes.hlsl, with AIR_PT_VOLUME): paths with kFlagProbe gather the light that
// arrives at a point in a volume through the volumes, except what the march adds itself: the sunlight
// and the light of its light groups that reach the point without scattering on the way.
#ifndef AIR_PT_PROBE
#define AIR_PT_PROBE 0
#endif
// Neural texture sets decoded on sample (SceneAccess.hlsli).
#ifndef AIR_PT_NTC
#define AIR_PT_NTC 0
#endif

struct IntegratorParams
{
    SkyData sky;      // the sky's radiance (a table or uniform) and the sun's disc
    CloudData clouds; // enabled == 0: none
    uint volumeCount; // the scene's volumes, read with LoadVolume(volumeTable, i)
    uint volumeTable; // bindless StructuredBuffer<VolumeData> on the GPU
    uint textureSets; // the materials' textures: bindless StructuredBuffer<TextureSetData> on the GPU
    uint textureSeed; // the frame's: neural texture sets filter stochastically
    uint lightCount;
    uint maxBounces;
    uint flags;            // kFlagFurnace, kFlagNoNee
    uint lightTreeBounces; // vertices 0 .. lightTreeBounces - 1 choose lights with the light tree
    uint cloudEvents;      // scattering events a walk through the clouds may take
    uint volumeEvents;     // scattering events a path may take in the volume, besides its bounces
    float nrcSpread;       // kFlagNrc: NrcData::spread
    uint nrcSlot;          // kFlagNrcTrain: the path's first training record
    uint nrcRecords;       // training records per path
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

// The indirect light of a path, as ReSTIR GI (ReSTIRGI.hlsli) resamples it: the part of the path
// beyond its second vertex x2 (the hit of the first surface's BSDF ray). From x2 it goes on in
// `direction`, either to the light NEE chose there (strategy 0) or along x2's BSDF sample
// (strategy 1). The radiance arriving at x2 along it is radianceMis (emission at the end of that
// segment, weighted against the other technique by the power heuristic, whose pdfs depend on the
// direction x2 is seen from) plus radianceFixed (everything further). One of the two candidates is
// chosen by RIS; `weight` is its unbiased contribution weight W in area measure at x2 times solid
// angle at x2.
struct GiPath
{
    bool valid;
    uint instance; // x2
    uint primitive;
    float2 barycentrics;
    float3 direction;
    uint strategy;
    float3 radianceMis;
    float3 radianceFixed;
    float lightPdf; // solid angle pdf of NEE at x2 choosing the light at the end of `direction`; 0: none
    float weight;
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
    // The first hit as TraceClosest reported it (instance kInvalidIndex: none), for ReSTIR.
    uint instance;
    uint primitive;
    float2 barycentrics;
    GiPath gi; // with kFlagRecordGi
    // With kFlagSignals, for denoisers: the light the first vertex reflects, split by the lobe that
    // reflects it (its own emission, and the sky a miss sees, are in neither), and how far its
    // reflected ray went (0: none).
    float3 diffuseRadiance;
    float3 specularRadiance;
    float reflectionDistance;
};

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

// What LoadSurface reads of a material's textures. A neural texture set (TextureSetData::latents) is
// decoded on sample only for the detail of the surfaces the camera sees; deeper vertices read its
// coarse copy (a quarter of its resolution, NeuralTextures.h), as their wide footprints would choose a
// coarse level anyway. Other sets read the same textures either way.
AIR_CONST uint kSurfaceGeometry = 0u; // none: the material's constants and the interpolated normal
AIR_CONST uint kSurfaceDetail = 1u;
AIR_CONST uint kSurfaceCoarse = 2u;

Surface LoadSurface(HitInfo hit, float3 rayDirection, IntegratorParams params, uint textures)
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
    if (m.textureSet != kInvalidIndex && textures != kSurfaceGeometry)
    {
        // The textures at the hit's texture coordinates (level 0: a path has no footprint to filter by).
        float2 uv0 = float2(v0.u, v0.v) * m.uvScale;
        float2 uv1 = float2(v1.u, v1.v) * m.uvScale;
        float2 uv2 = float2(v2.u, v2.v) * m.uvScale;
        TextureSetData set = LoadTextureSet(params.textureSets, m.textureSet);
        float2 uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
        float4 color;
        float4 surface;
#if AIR_PT_NTC
        if (textures == kSurfaceDetail && set.latents != kInvalidIndex)
        {
            // Stochastic filtering's choice by the hit and the frame: the same wherever the frame loads
            // the hit again (ReSTIR's passes), different every frame.
            uint h = Pcg(asuint(hit.barycentrics.x) ^ Pcg(asuint(hit.barycentrics.y) ^
                                                          Pcg(hit.primitive ^ Pcg(hit.instance + params.textureSeed))));
            float2 u = float2(float(h & 0xFFFFu), float(h >> 16)) / 65536.0f;
            NeuralTextureSample(set, uv, u, color, surface);
        }
        else
#endif
        {
            color = MaterialTextureSample(set.color, uv);
            surface = MaterialTextureSample(set.surface, uv);
        }
        s.material.baseColor *= float3(color.x, color.y, color.z);
        s.material.roughness = saturate(s.material.roughness * surface.z);
        s.material.metallic = saturate(s.material.metallic * surface.w);
        // The normal map in the triangle's tangent frame (along increasing u and v), around the
        // shading normal.
        float2 d1 = uv1 - uv0;
        float2 d2 = uv2 - uv0;
        float det = d1.x * d2.y - d1.y * d2.x;
        float2 bend = float2(surface.x, surface.y) * 2.0f - 1.0f;
        if (abs(det) > 1e-12f && (bend.x != 0.0f || bend.y != 0.0f))
        {
            float3 tangent = ((p1 - p0) * d2.y - (p2 - p0) * d1.y) / det;
            float3 bitangent = ((p2 - p0) * d1.x - (p1 - p0) * d2.x) / det;
            float3 n = s.shadingNormal;
            float3 t = tangent - n * dot(n, tangent);
            if (dot(t, t) > 1e-20f)
            {
                t = normalize(t);
                float3 b = cross(n, t);
                b = dot(b, bitangent) < 0.0f ? -b : b;
                float3 bent = normalize(t * bend.x + b * bend.y + n * sqrt(saturate(1.0f - dot(bend, bend))));
                if (dot(bent, s.geometricNormal) > 0.0f)
                {
                    s.shadingNormal = bent;
                }
            }
        }
    }
    s.emission = frontFace ? m.emission : float3(0.0f, 0.0f, 0.0f);
    if ((params.flags & kFlagFurnace) != 0u)
    {
        s.material.baseColor = float3(1.0f, 1.0f, 1.0f);
        s.emission = float3(0.0f, 0.0f, 0.0f);
    }
    s.lightIndex = any(s.emission) ? LoadInstanceLight(hit.instance) + hit.primitive : kInvalidIndex;
    return s;
}

// --- Clouds --------------------------------------------------------------------------------------

// Radiance arriving at `origin` along `direction` from the clouds and the sky beyond them, for a ray
// that meets no surface before tMax (1e30: none at all). A random walk by delta tracking: at every
// scattering event, next event estimation of the sun (its transmittance by ratio tracking) weighed
// by the power heuristic against phase sampling, which continues the walk. A walk that leaves the
// layer downwards sees the ground of the sky table: light the scene's own geometry would reflect
// back up into the clouds is not traced. `sun` is the disc seen straight through (the caller weighs
// it against its sun NEE); `rest` is everything else, the air in front of the clouds included.
// `surface`: the ray reached its surface at tMax without meeting the clouds, and nothing else was
// gathered.
struct Escape
{
    float3 sun;
    float3 rest;
    bool surface;
};

Escape EscapeRadiance(IntegratorParams params, float3 origin, float3 direction, float tMax, AIR_INOUT(uint) seed)
{
    Escape escape;
    escape.sun = float3(0.0f, 0.0f, 0.0f);
    escape.rest = float3(0.0f, 0.0f, 0.0f);
    escape.surface = false;
    CloudData c = params.clouds;
    float albedo = (params.flags & kFlagFurnace) != 0u ? 1.0f : c.albedo;
    bool sunNee = HasSun(params.sky) && (params.flags & (kFlagNoNee | kFlagFurnace)) == 0u;
    float sunPdf = sunNee ? 1.0f / SunSolidAngle(params.sky) : 0.0f;
    float throughput = 1.0f;
    float phasePdf = 0.0f; // of the phase sample that produced `d`
    float3 p = origin;
    float3 d = direction;
    Aerial air; // between the origin and the first collision
    air.inscatter = float3(0.0f, 0.0f, 0.0f);
    air.transmittance = float3(1.0f, 1.0f, 1.0f);
    for (uint event = 0u; event <= params.cloudEvents; ++event)
    {
        float t = 0.0f;
        if (!CloudCollision(c, p, d, event == 0u ? tMax : 1e30f, seed, t))
        {
            if (event == 0u && tMax < 1e30f)
            {
                escape.surface = true;
                break;
            }
            float3 sun = float3(0.0f, 0.0f, 0.0f);
            if (HasSun(params.sky) && dot(d, params.sky.sunDirection) >= params.sky.sunCosRadius)
            {
                sun = params.sky.sunRadiance;
            }
            float3 sky = SkyRadiance(params.sky, d);
            if (event == 0u)
            {
                escape.sun = sun;
                escape.rest = sky;
            }
            else
            {
                escape.rest += (sky + sun * (sunNee ? PowerHeuristic(phasePdf, sunPdf) : 1.0f)) * throughput;
            }
            break;
        }
        if (event == params.cloudEvents)
        {
            break; // the walk is too long: its light is lost (rare, see docs/architecture.md §8.10)
        }
        if (event == 0u)
        {
            air = AerialPerspective(params.sky, direction, t);
        }
        p = p + d * t;
        throughput *= albedo;
        if (sunNee)
        {
            float3 wi = SampleSunDirection(params.sky, Rand2(seed));
            float phase = CloudPhase(c, dot(d, wi));
            float transmittance = CloudTransmittance(c, p, wi, 1e30f, seed);
            float weight = PowerHeuristic(sunPdf, phase);
            escape.rest += params.sky.sunRadiance * (throughput * transmittance * phase * weight / sunPdf);
        }
        float3 u = Rand3(seed);
        float3 next = SampleCloudPhase(c, d, u);
        phasePdf = CloudPhase(c, dot(d, next));
        d = next;
        if (throughput < 0.25f)
        {
            if (Rand(seed) >= throughput * 4.0f)
            {
                break;
            }
            throughput = 0.25f;
        }
    }
    escape.rest = air.inscatter + air.transmittance * escape.rest;
    return escape;
}

// The air between the camera and the surface its ray hit: what it adds and what it lets through of
// the path's light. ReSTIR's light at that surface goes through the same air (ReSTIRCommon.hlsli).
void AddAerialPerspective(IntegratorParams params, float3 direction, AIR_INOUT(PathSample) s)
{
    if (s.hit)
    {
        Aerial air = AerialPerspective(params.sky, direction, s.hitDistance);
        s.radiance = air.inscatter + air.transmittance * s.radiance;
        s.directRadiance = air.inscatter + air.transmittance * s.directRadiance;
        s.diffuseRadiance *= air.transmittance;
        s.specularRadiance *= air.transmittance;
    }
}

// --- Neural radiance cache -----------------------------------------------------------------------

#if AIR_PT_NRC
// The cache's view of a vertex the light leaves towards wo.
NrcQuery NrcMakeQuery(Surface s, float3 wo)
{
    float3 n = dot(s.shadingNormal, wo) > 0.0f ? s.shadingNormal : s.geometricNormal;
    NrcQuery q;
    q.position = s.position;
    q.normal = NrcPackDirection(n);
    q.direction = NrcPackDirection(wo);
    q.roughness = NrcPackUnorm16x2(s.material.roughness, 0.0f);
    q.diffuse = NrcPackColor(DiffuseAlbedo(s.material));
    q.specular = NrcPackColor(SpecularAlbedo(s.material, saturate(dot(n, wo))));
    return q;
}
#endif

#if AIR_PT_VOLUME
// The nearest real collision with any of the volumes along o + t d within [0, tMax], and the volume
// it is in. Each volume is tracked only up to the nearest collision found so far.
bool VolumesCollision(IntegratorParams params, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed,
                      AIR_OUT(float) tHit, AIR_OUT(VolumeData) hitVolume)
{
    tHit = tMax;
    hitVolume = LoadVolume(params.volumeTable, 0u);
    bool found = false;
    for (uint i = 0u; i < params.volumeCount; ++i)
    {
        VolumeData v = LoadVolume(params.volumeTable, i);
        float t;
        if (VolumeCollision(v, o, d, tHit, seed, t))
        {
            tHit = t;
            hitVolume = v;
            found = true;
        }
    }
    return found;
}

// The transmittance through every volume along o + t d over [0, tMax]: the product of each one's
// estimate.
float VolumesTransmittance(IntegratorParams params, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed)
{
    float transmittance = 1.0f;
    for (uint i = 0u; i < params.volumeCount && transmittance > 0.0f; ++i)
    {
        transmittance *= VolumeTransmittance(LoadVolume(params.volumeTable, i), o, d, tMax, seed);
    }
    return transmittance;
}
#endif

// The volumes' transmittance along a shadow ray from a surface towards a triangle light, and towards
// the sun: tracked, from their moment shadow maps, or 1 in kernels without volumes.
float LightTransmittance(IntegratorParams params, LightTriangle light, float3 origin, float3 direction, float tMax,
                         AIR_INOUT(uint) seed)
{
#if AIR_PT_VOLUME
    return VolumesTransmittance(params, origin, direction, tMax, seed);
#elif AIR_PT_VOLUME_SHADOWS
    return VolumeGroupShadow(light.group, origin, origin + direction * tMax);
#else
    return 1.0f;
#endif
}

float SunTransmittance(IntegratorParams params, float3 origin, float3 direction, AIR_INOUT(uint) seed)
{
#if AIR_PT_VOLUME
    return VolumesTransmittance(params, origin, direction, 1e30f, seed);
#elif AIR_PT_VOLUME_SHADOWS
    return VolumeSunShadow(origin);
#else
    return 1.0f;
#endif
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
    result.gi.valid = false;
    result.gi.instance = kInvalidIndex;
    result.gi.primitive = 0u;
    result.gi.barycentrics = float2(0.0f, 0.0f);
    result.gi.direction = float3(0.0f, 0.0f, 1.0f);
    result.gi.strategy = 0u;
    result.gi.radianceMis = float3(0.0f, 0.0f, 0.0f);
    result.gi.radianceFixed = float3(0.0f, 0.0f, 0.0f);
    result.gi.lightPdf = 0.0f;
    result.gi.weight = 0.0f;
    result.diffuseRadiance = float3(0.0f, 0.0f, 0.0f);
    result.specularRadiance = float3(0.0f, 0.0f, 0.0f);
    result.reflectionDistance = 0.0f;
    // The split: the first vertex's next event estimation goes by its BSDF's parts, the rest of the
    // path (from `reflectedFrom` on) by those of the BSDF sample it continued with.
    bool signals = (params.flags & kFlagSignals) != 0u;
    bool reflected = false;
    float3 reflectedFrom = float3(0.0f, 0.0f, 0.0f);
    float3 diffuseShare = float3(0.0f, 0.0f, 0.0f);

    bool useNee = params.lightCount > 0u && (params.flags & kFlagNoNee) == 0u && (params.flags & kFlagFurnace) == 0u;
    bool sunNee = HasSun(params.sky) && (params.flags & (kFlagNoNee | kFlagFurnace)) == 0u;
    float sunPdf = sunNee ? 1.0f / SunSolidAngle(params.sky) : 0.0f; // of sun NEE, per solid angle
    // ReSTIR DI estimates the whole direct light at the first surface (from emissive triangles): the
    // path neither samples a light there nor counts the emission its BSDF sample reaches.
    bool restir = useNee && (params.flags & kFlagReSTIR) != 0u;
    // ReSTIR GI estimates the light of every path with two or more bounces, so none of it goes into
    // `radiance`. The sample that records it (kFlagRecordGi) gathers it as candidates at x2: NEE
    // there (a triangle and the sun, streamed into one by weighted reservoir sampling), and the BSDF
    // continuation, whose incoming radiance splits into the emission it reaches (triangle or sun) and
    // the rest. Other samples stop at x2.
    bool gi = AIR_PT_RESTIR_GI != 0 && (params.flags & kFlagReSTIRGI) != 0u;
    bool recordGi = gi && (params.flags & kFlagRecordGi) != 0u;
    float firstBsdfPdf = 0.0f; // solid angle pdf of the ray from the first surface to x2
    float x2AreaPdf = 0.0f;
    float3 x2Throughput = float3(0.0f, 0.0f, 0.0f);
    float neeWeightSum = 0.0f; // candidate 0, NEE at x2: luminances of the contributions streamed
    float neeWeight = 0.0f;    // of the one kept
    float3 neeDirection = float3(0.0f, 0.0f, 1.0f);
    float3 neeRadiance = float3(0.0f, 0.0f, 0.0f);
    float neeLightPdf = 0.0f;
    bool continued = false; // candidate 1: the BSDF continuation from x2
    float3 continuationDirection = float3(0.0f, 0.0f, 1.0f);
    float continuationPdf = 0.0f;
    float3 continuationWeight = float3(0.0f, 0.0f, 0.0f);
    float3 continuationEmission = float3(0.0f, 0.0f, 0.0f);
    float continuationLightPdf = 0.0f;
    float3 continuationRest = float3(0.0f, 0.0f, 0.0f);
    float3 restThroughput = float3(1.0f, 1.0f, 1.0f); // from x3 on, relative to the radiance reaching x2

#if AIR_PT_NRC
    bool nrc = (params.flags & kFlagNrc) != 0u;
    bool nrcTrain = nrc && (params.flags & kFlagNrcTrain) != 0u;
    bool nrcUnbiased = (params.flags & kFlagNrcUnbiased) != 0u;
    float nrcBudget = 0.0f;       // nrcSpread times the primary vertex's spread
    float nrcSpreadRoot = 0.0f;   // sum of the square roots of the segments' spreads since then
    float nrcSuffixRoot = 0.0f;   // the same since a training path's render query
    bool nrcRenderDone = false;   // the pixel's radiance is complete (a training path goes on)
    float3 nrcRenderRadiance = float3(0.0f, 0.0f, 0.0f);
    float3 nrcRenderDirect = float3(0.0f, 0.0f, 0.0f);
    bool nrcHasQuery = false; // the path ends in the cache: at nrcQuery, times nrcQueryThroughput
    NrcQuery nrcQuery;
    nrcQuery.position = float3(0.0f, 0.0f, 0.0f);
    nrcQuery.normal = 0u;
    nrcQuery.direction = 0u;
    nrcQuery.roughness = 0u;
    nrcQuery.diffuse = 0u;
    nrcQuery.specular = 0u;
    float3 nrcQueryThroughput = float3(0.0f, 0.0f, 0.0f);
    uint nrcRecordCount = 0u;
#endif

    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float previousBsdfPdf = 0.0f; // solid angle pdf of the BSDF sample that produced `direction`
    // The vertex that produced `direction` and how NEE chose lights there (for MIS).
    float3 previousPosition = origin;
    float3 previousNormal = float3(0.0f, 0.0f, 0.0f);
    bool previousTree = false;
    uint events = 0u; // scattering events in the volumes so far; `bounce` counts them as vertices

    for (uint bounce = 0u; bounce <= params.maxBounces + events; ++bounce)
    {
        HitInfo hit = TraceClosest(origin, direction, bounce);
        bool direct = bounce <= 1u;
        if (bounce == 1u && reflected)
        {
            result.reflectionDistance = hit.hit ? hit.t : kMissReflectionDistance;
        }
        bool beyondX2 = gi && bounce >= 2u; // light found here is the continuation candidate's

#if AIR_PT_VOLUME
        // The volumes: a real collision before the surface (or before the ray leaves the scene)
        // scatters the path there, as the volume it is in does.
        float tVolume = 0.0f;
        VolumeData volume;
        if (params.volumeCount > 0u &&
            VolumesCollision(params, origin, direction, hit.hit ? hit.t : 1e30f, seed, tVolume, volume))
        {
            float3 p = origin + direction * tVolume;
            float albedo = (params.flags & kFlagFurnace) != 0u ? 1.0f : volume.albedo;
            if (bounce == 0u)
            {
                result.hit = true;
                result.hitDistance = tVolume;
                result.position = p;
                result.normal = -direction;
                result.material.baseColor = float3(albedo, albedo, albedo);
                result.material.roughness = 1.0f;
                result.material.metallic = 0.0f;
                result.diffuseAlbedo = result.material.baseColor;
            }
            if (events == params.volumeEvents)
            {
                break; // the walk is too long: its light is lost
            }
            ++events;
            result.bounces = bounce + 1u;
            throughput *= albedo;
            bool lightTree = bounce < params.lightTreeBounces;
            if (useNee)
            {
                // One point on one emissive triangle, through the volumes and past the surfaces.
                float uLight = Rand(seed);
                float uAlias = Rand(seed);
                float2 uPoint = Rand2(seed);
                LightTriangle light = LoadLight(0u);
                float pmf = 0.0f;
                if (lightTree)
                {
                    LightChoice choice = SampleLightTree(p, float3(0.0f, 0.0f, 0.0f), uLight);
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
                float3 toLight = light.p0 + light.e1 * b.x + light.e2 * b.y - p;
                float dist = length(toLight);
                float3 wi = toLight / dist;
                float cosLight = dot(-wi, normalize(cross(light.e1, light.e2)));
                float lightPdf = pmf > 0.0f ? LightPdf(pmf, light.area, dist, cosLight) : 0.0f;
                if (lightPdf > 0.0f && TraceVisible(p, wi, dist * (1.0f - 1e-4f)))
                {
                    float phase = VolumePhase(volume, dot(direction, wi));
                    float transmittance = VolumesTransmittance(params, p, wi, dist, seed);
                    float3 contribution =
                        throughput * light.radiance * (phase * transmittance * PowerHeuristic(lightPdf, phase) / lightPdf);
                    result.radiance += contribution;
                    if (bounce == 0u)
                    {
                        result.directRadiance += contribution;
                        result.diffuseRadiance += contribution;
                    }
                }
            }
            if (sunNee)
            {
                float3 wi = SampleSunDirection(params.sky, Rand2(seed));
                if (TraceVisible(p, wi, 1e30f))
                {
                    float phase = VolumePhase(volume, dot(direction, wi));
                    // One estimate after the other: the operands of * are unsequenced in C++.
                    float transmittance = VolumesTransmittance(params, p, wi, 1e30f, seed);
                    transmittance *= CloudTransmittance(params.clouds, p, wi, 1e30f, seed);
                    float3 contribution = throughput * params.sky.sunRadiance *
                                          (phase * transmittance * PowerHeuristic(sunPdf, phase) / sunPdf);
                    result.radiance += contribution;
                    if (bounce == 0u)
                    {
                        result.directRadiance += contribution;
                        result.diffuseRadiance += contribution;
                    }
                }
            }
            // Continue along a direction the phase function chose.
            float3 next = SampleVolumePhase(volume, direction, Rand2(seed));
            if (bounce == 0u)
            {
                reflected = true; // a medium reflects like a diffuse surface
                reflectedFrom = result.radiance;
                diffuseShare = float3(1.0f, 1.0f, 1.0f);
            }
            previousBsdfPdf = VolumePhase(volume, dot(direction, next));
            previousPosition = p;
            previousNormal = float3(0.0f, 0.0f, 0.0f);
            previousTree = lightTree;
            origin = p;
            direction = next;
            float survive = MaxComponent(throughput) * 4.0f;
            if (survive < 1.0f)
            {
                if (Rand(seed) >= survive)
                {
                    break;
                }
                throughput /= survive;
            }
            continue;
        }
#endif
        // Misses go on through the clouds; a ray from above their base (only the camera's) may meet
        // them before its surface.
        Escape escape;
        escape.sun = float3(0.0f, 0.0f, 0.0f);
        escape.rest = float3(0.0f, 0.0f, 0.0f);
        escape.surface = true;
        if (!hit.hit || (params.clouds.enabled != 0u && origin.y > params.clouds.bottom))
        {
            escape = EscapeRadiance(params, origin, direction, hit.hit ? hit.t : 1e30f, seed);
        }
        if (!hit.hit || !escape.surface)
        {
            // The sky and the clouds, and the sun when the ray reaches it within its disc, weighed
            // against sun NEE at the vertex the ray left.
            float3 sky = escape.rest;
            float3 sun = escape.sun;
            if (beyondX2 && bounce == 2u)
            {
                // Reached from x2: ReSTIR GI weighs the sun itself (see the emission below).
                continuationEmission = sun;
                continuationLightPdf = sunPdf;
                continuationRest += sky;
            }
            else
            {
                float sunWeight = bounce > 0u ? PowerHeuristic(previousBsdfPdf, sunPdf) : 1.0f;
#if AIR_PT_PROBE
                if (bounce == 0u && (params.flags & kFlagProbe) != 0u)
                {
                    sunWeight = 0.0f; // a probe's direct sunlight: the march's own
                }
#endif
                float3 environment = sky + sun * sunWeight;
                if (beyondX2)
                {
                    continuationRest += restThroughput * environment;
                }
                else
                {
                    result.radiance += throughput * environment;
                    if (direct)
                    {
                        result.directRadiance += throughput * environment;
                    }
                }
            }
            break;
        }
        Surface s = LoadSurface(hit, direction, params, bounce == 0u ? kSurfaceDetail : kSurfaceCoarse);
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
        if (bounce == 1u && recordGi)
        {
            result.gi.instance = hit.instance;
            result.gi.primitive = hit.primitive;
            result.gi.barycentrics = hit.barycentrics;
            x2AreaPdf = firstBsdfPdf * abs(dot(s.geometricNormal, direction)) / (hit.t * hit.t);
            x2Throughput = throughput;
        }

        // Emission found by the BSDF sample, weighted against NEE having found it.
        bool emits = any(s.emission) && !(restir && bounce == 1u);
#if AIR_PT_PROBE
        if (emits && bounce == 0u && (params.flags & kFlagProbe) != 0u)
        {
            emits = !VolumeExplicitGroup(LoadLight(s.lightIndex).group); // the march's own light
        }
#endif
        if (emits)
        {
            float weight = 1.0f;
            float lightPdf = 0.0f;
            if (bounce > 0u && useNee)
            {
                LightTriangle light = LoadLight(s.lightIndex);
                float pmf = previousTree ? LightTreePmf(s.lightIndex, previousPosition, previousNormal) : light.probability;
                lightPdf = LightPdf(pmf, light.area, hit.t, dot(-direction, s.geometricNormal));
                weight = PowerHeuristic(previousBsdfPdf, lightPdf);
            }
            if (beyondX2 && bounce == 2u)
            {
                // Reached from x2: the MIS weight depends on the direction x2 is seen from, which
                // ReSTIR GI changes, so it keeps the unweighted emission and the light pdf.
                continuationEmission = s.emission;
                continuationLightPdf = lightPdf;
            }
            else if (beyondX2)
            {
                continuationRest += restThroughput * s.emission * weight;
            }
            else
            {
                float3 emitted = throughput * s.emission * weight;
                result.radiance += emitted;
                if (direct)
                {
                    result.directRadiance += emitted;
                }
            }
        }
#if AIR_PT_NRC
        if (nrc)
        {
            // The footprint's spread (Mueller et al. 2021, section 3.4): at the primary vertex as if the
            // camera emitted uniformly in all directions, beyond it the square of the sum of every
            // segment's square root.
            float cosHit = max(abs(dot(direction, s.geometricNormal)), 1e-4f);
            if (bounce == 0u)
            {
                nrcBudget = params.nrcSpread * hit.t * hit.t / (4.0f * kPi * cosHit);
            }
            else
            {
                float root = sqrt(hit.t * hit.t / (max(previousBsdfPdf, 1e-12f) * cosHit));
                nrcSpreadRoot += root;
                nrcSuffixRoot += root;
            }
            bool last = bounce == params.maxBounces + events;
            float3 wo = -direction;
            bool tail = false;
            if (!nrcRenderDone && (last || (bounce > 0u && nrcSpreadRoot * nrcSpreadRoot > nrcBudget)))
            {
                nrcRenderDone = true;
                if (!nrcTrain)
                {
                    nrcQuery = NrcMakeQuery(s, wo);
                    nrcQueryThroughput = throughput;
                    nrcHasQuery = true;
                    break;
                }
                NrcDeferQuery(NrcMakeQuery(s, wo), throughput);
                nrcRenderRadiance = result.radiance;
                nrcRenderDirect = result.directRadiance;
                nrcSuffixRoot = 0.0f;
                tail = last;
            }
            else if (nrcRenderDone && (last || (!nrcUnbiased && nrcSuffixRoot * nrcSuffixRoot > nrcBudget)))
            {
                tail = true;
            }
            if (tail)
            {
                // The training suffix ends in the cache too (self-training).
                nrcQuery = NrcMakeQuery(s, wo);
                nrcQueryThroughput = throughput;
                nrcHasQuery = true;
                break;
            }
            // What this vertex reflects is a training sample (at the first vertex only when its
            // direct light is part of the path, not ReSTIR DI's).
            if (nrcTrain && nrcRecordCount < params.nrcRecords && (bounce > 0u || !restir))
            {
                NrcTrainingRecord record;
                record.query = NrcMakeQuery(s, wo);
                record.value = result.radiance;
                record.valid = 0u;
                record.throughput = throughput;
                record.pad = 0u;
                NrcWriteRecord(params.nrcSlot + nrcRecordCount, record);
                ++nrcRecordCount;
            }
        }
#endif
        if (bounce == params.maxBounces + events || (gi && !recordGi && bounce == 1u))
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
            float2 uPoint = Rand2(seed);
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
                    float weight =
                        PowerHeuristic(lightPdf, e.pdf) * LightTransmittance(params, light, rayOrigin, wi, dist, seed);
                    float3 contribution = throughput * e.fCos * light.radiance * (weight / lightPdf);
                    if (gi && bounce == 1u)
                    {
                        float w = Luminance(contribution);
                        neeWeightSum += w;
                        if (Rand(seed) * neeWeightSum < w)
                        {
                            neeWeight = w;
                            neeDirection = wi;
                            neeRadiance = light.radiance;
                            neeLightPdf = lightPdf;
                        }
                    }
                    else if (beyondX2)
                    {
                        continuationRest += restThroughput * e.fCos * light.radiance * (weight / lightPdf);
                    }
                    else
                    {
                        result.radiance += contribution;
                        if (bounce == 0u)
                        {
                            result.directRadiance += contribution;
                            float3 diffuse = throughput * e.fCosDiffuse * light.radiance * (weight / lightPdf);
                            result.diffuseRadiance += diffuse;
                            result.specularRadiance += contribution - diffuse;
                        }
                    }
                }
            }
        }

        // The sun: one direction within its disc, through the clouds. (ReSTIR DI only takes over
        // the triangles.)
        if (sunNee)
        {
            float2 uSun = Rand2(seed);
            float3 wi = SampleSunDirection(params.sky, uSun);
            if (dot(wi, ng) > 0.0f)
            {
                BsdfEval e = EvaluateBsdf(s.material, woLocal, ToLocal(frame, wi));
                if (any(e.fCos) && TraceVisible(rayOrigin, wi, 1e30f))
                {
                    // One estimate after the other: the operands of * are unsequenced in C++.
                    float transmittance = CloudTransmittance(params.clouds, rayOrigin, wi, 1e30f, seed);
                    transmittance *= SunTransmittance(params, rayOrigin, wi, seed);
                    float3 sunRadiance = params.sky.sunRadiance * transmittance;
                    float weight = PowerHeuristic(sunPdf, e.pdf);
                    float3 contribution = throughput * e.fCos * sunRadiance * (weight / sunPdf);
                    if (gi && bounce == 1u)
                    {
                        float w = Luminance(contribution);
                        neeWeightSum += w;
                        if (Rand(seed) * neeWeightSum < w)
                        {
                            neeWeight = w;
                            neeDirection = wi;
                            neeRadiance = sunRadiance;
                            neeLightPdf = sunPdf;
                        }
                    }
                    else if (beyondX2)
                    {
                        continuationRest += restThroughput * e.fCos * sunRadiance * (weight / sunPdf);
                    }
                    else
                    {
                        result.radiance += contribution;
                        if (bounce == 0u)
                        {
                            result.directRadiance += contribution;
                            float3 diffuse = throughput * e.fCosDiffuse * sunRadiance * (weight / sunPdf);
                            result.diffuseRadiance += diffuse;
                            result.specularRadiance += contribution - diffuse;
                        }
                    }
                }
            }
        }

        // Continue the path by sampling the BSDF.
        float uLobe = Rand(seed);
        float2 u = Rand2(seed);
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
        if (bounce == 0u)
        {
            reflected = true;
            reflectedFrom = result.radiance;
            // diffuseWeight is 0 wherever weight is.
            diffuseShare = bs.diffuseWeight / max(bs.weight, float3(1e-30f, 1e-30f, 1e-30f));
        }
        if (gi && bounce == 0u)
        {
            firstBsdfPdf = bs.pdf;
        }
        else if (gi && bounce == 1u)
        {
            continued = true;
            continuationDirection = wiWorld;
            continuationPdf = bs.pdf;
            continuationWeight = bs.weight;
        }
        else if (beyondX2)
        {
            restThroughput *= bs.weight;
        }
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
            restThroughput /= survive;
        }
    }

    // ReSTIR GI: RIS between the candidates at x2 by the luminance of what each contributes (the NEE
    // ones were already streamed into one, with their weight sum).
    if (recordGi && x2AreaPdf > 0.0f)
    {
        float continuationMis = continuationLightPdf > 0.0f ? PowerHeuristic(continuationPdf, continuationLightPdf) : 1.0f;
        float3 continuationContribution = float3(0.0f, 0.0f, 0.0f);
        if (continued)
        {
            continuationContribution =
                x2Throughput * continuationWeight * (continuationEmission * continuationMis + continuationRest);
        }
        float wContinuation = Luminance(continuationContribution);
        float total = neeWeightSum + wContinuation;
        if (total > 0.0f)
        {
            // W = sum(w) / p̂(Y), p̂(Y) being the luminance of the integrand: contribution * pdf(Y).
            bool nee = Rand(seed) * total < neeWeightSum;
            result.gi.valid = true;
            if (nee)
            {
                result.gi.strategy = 0u;
                result.gi.direction = neeDirection;
                result.gi.radianceMis = neeRadiance;
                result.gi.lightPdf = neeLightPdf;
                result.gi.weight = total / (neeWeight * x2AreaPdf * neeLightPdf);
            }
            else
            {
                result.gi.strategy = 1u;
                result.gi.direction = continuationDirection;
                result.gi.radianceMis = continuationEmission;
                result.gi.radianceFixed = continuationRest;
                result.gi.lightPdf = continuationLightPdf;
                result.gi.weight = total / (wContinuation * x2AreaPdf * continuationPdf);
            }
        }
    }
#if AIR_PT_NRC
    // Every thread predicts, those without a query on a dummy one: the matrix units then see whole
    // waves, and the linear algebra driver tested here returned NaN for some threads of a call made
    // only by the threads with a query (after this loop's breaks).
    float3 nrcPrediction = NrcPredict(nrcQuery);
    if (nrcHasQuery)
    {
        result.radiance += nrcQueryThroughput * nrcPrediction;
    }
    if (nrcTrain)
    {
        // Every record's target: what the rest of the path found (the cache's prediction at its end
        // included) per unit of throughput up to the vertex. Records the path did not reach are
        // marked invalid (the slots hold an earlier frame's).
        for (uint i = 0u; i < params.nrcRecords; ++i)
        {
            NrcTrainingRecord record = NrcReadRecord(params.nrcSlot + i);
            float3 t = record.throughput;
            bool valid = i < nrcRecordCount && min(t.x, min(t.y, t.z)) > 1e-12f;
            float3 reflected = result.radiance - record.value;
            record.value = valid ? float3(max(reflected.x, 0.0f) / t.x, max(reflected.y, 0.0f) / t.y,
                                          max(reflected.z, 0.0f) / t.z)
                                 : float3(0.0f, 0.0f, 0.0f);
            record.valid = valid ? 1u : 0u;
            NrcWriteRecord(params.nrcSlot + i, record);
        }
        if (nrcRenderDone)
        {
            result.radiance = nrcRenderRadiance;
            result.directRadiance = nrcRenderDirect;
        }
    }
#endif
    if (signals && reflected)
    {
        float3 rest = result.radiance - reflectedFrom;
        result.diffuseRadiance += rest * diffuseShare;
        result.specularRadiance += rest * (1.0f - diffuseShare);
    }
    return result;
}

#endif
