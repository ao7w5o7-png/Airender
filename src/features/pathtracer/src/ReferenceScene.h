#pragma once

// The CPU renderers' scene (ReferenceRenderer.cpp, VolumeMarchReference.cpp): the GPU's tables in
// std::vectors, a median-split BVH over its triangles, and the hooks through which the shader code
// they share with the GPU (Integrator.hlsli and what it includes) reads them. Each renderer includes
// that code into a kernel struct of its own, derived from SceneHooks, where the shader functions
// become member functions; kernels of different AIR_PT_* configurations live in translation units of
// their own (the shader files have include guards).

#include "air/features/pathtracer/Bsdf.h"
#include "air/features/pathtracer/ReferenceRenderer.h"
#include "air/renderer/Sky.h"
#include "air/renderer/TextureSampling.h"
#include "air/renderer/Volume.h"
#include "air/renderer/shader/SceneShared.h"
#include "air/renderer/shader/SkyShared.h"
#include "air/renderer/shader/VolumeShared.h"

#include "PathTracerShared.h"

#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace air::pt::reference
{

using namespace air::hlsl;
using namespace air::gpu;
using namespace air::pt::gpu;

// --- BVH over world-space triangles --------------------------------------------------------------

struct Triangle
{
    float3 p0, e1, e2;
    u32 instance;
    u32 primitive;
    bool cutout; // its material is a cutout: hits go through the alpha test (Cutout.hlsli)
};

// The alpha test of a cutout's triangle: true where the ray passes (AlphaCutsOut).
using CutoutTest = std::function<bool(u32 instance, u32 primitive, float2 barycentrics)>;

struct Bounds
{
    float3 lo = float3(1e30f);
    float3 hi = float3(-1e30f);
    void Grow(float3 p)
    {
        lo = min(lo, p);
        hi = max(hi, p);
    }
};

struct BvhNode
{
    Bounds bounds;
    u32 first = 0; // leaf: first triangle; inner: index of the left child (right = first + 1)
    u32 count = 0; // > 0: leaf
};

class Bvh
{
public:
    explicit Bvh(std::vector<Triangle> triangles);

    // The closest hit with t in (tMin, tMax), past the triangles of instance `skip` and, with `cutout`,
    // past the holes of cutouts. anyHit: stop at the first hit.
    bool Intersect(float3 origin, float3 direction, float tMax, bool anyHit, HitInfo& hit, float tMin = 0.0f,
                   u32 skip = kInvalidIndex, const CutoutTest* cutout = nullptr) const;

private:
    void Build(u32 nodeIndex, u32 begin, u32 end);
    static bool HitsBox(const Bounds& b, float3 origin, float3 inv, float tMax);
    static bool IntersectTriangle(const Triangle& tri, float3 origin, float3 direction, float tMin, HitInfo& hit,
                                  const CutoutTest* cutout);

    std::vector<Triangle> mTriangles;
    std::vector<BvhNode> mNodes;
};

// --- Scene tables --------------------------------------------------------------------------------

struct SceneTables
{
    std::vector<InstanceData> instances; // vertexBuffer / indexBuffer = mesh index
    std::vector<u32> instanceLights;     // first light-list entry per instance
    std::vector<std::span<const Vertex>> vertices;
    std::vector<std::span<const u32>> indices;
    std::vector<MaterialData> materials;
    // The materials' textures; TextureSetData::color / surface of set i are 2 i and 2 i + 1.
    std::vector<const renderer::MaterialTextures*> textureSets;
    std::vector<LightTriangle> lights; // at least one entry, like the GPU list
    std::vector<LightTreeNode> lightTree;
    std::vector<LightGroup> lightGroups; // one per emissive instance
    double lightPower = 0.0;             // 0: no emitters (`lights` holds a placeholder)
    std::span<const float> lut;
    std::vector<float> neuralCoat; // the coated materials' network (NeuralCoat.h); empty: none
    renderer::SkyTable sky;
    renderer::CloudVolume clouds;
    // The volumes that loaded; their VolumeData's density and majorants hold their index here.
    std::vector<renderer::VolumeResources> volumes;
    // The real-time volumes' moment shadow map of each light group (VolumeShadowLayout), for their
    // probes (VolumeExplicitGroup); empty: none.
    std::vector<u32> groupMaps;
    std::unique_ptr<Bvh> bvh;
};

// The tables of a scene that passes renderer::Validate, its BVH included, and the neural coat of its
// coated materials (trained, or found in the cache, with `tasks` when set).
SceneTables BuildSceneTables(const renderer::SceneDesc& scene, TaskSystem* tasks = nullptr);

CameraData ToCameraData(const renderer::Camera& camera, float aspect);

// The sky the integrator sees and the scale from its radiance to the image: the camera's exposure, or
// in the furnace (as the GPU path tracer has it) a uniform sky and no sun, mapped to 0.18.
struct ReferenceLighting
{
    SkyData sky;
    float exposure = 1.0f;
};

ReferenceLighting LightingFor(const SceneTables& tables, const renderer::Camera& camera,
                              const ReferenceSettings& settings);

// A kernel's IntegratorParams for the settings, with `flags` besides the furnace and NEE's.
template <typename Params>
Params MakeParams(const SceneTables& tables, const ReferenceSettings& settings, const SkyData& sky, u32 flags)
{
    Params params{};
    params.sky = sky;
    params.clouds = tables.clouds.data;
    params.volumeCount = u32(tables.volumes.size());
    params.volumeTable = 0;
    params.textureSets = 0;
    params.textureSeed = 0; // the CPU renders neural sets from their decoded texels
    // Without a network (no coated materials) a neural coat is the approximation.
    params.coatModel = tables.neuralCoat.empty() && settings.coatModel == u32(CoatModel::Neural)
                           ? u32(CoatModel::Approximate)
                           : settings.coatModel;
    params.cloudEvents = settings.cloudEvents;
    params.volumeEvents = settings.volumeEvents;
    params.lightCount = tables.lightPower > 0.0 ? u32(tables.lights.size()) : 0u;
    params.maxBounces = settings.maxBounces;
    params.flags = flags | (settings.furnace ? kFlagFurnace : 0u) | (settings.nextEventEstimation ? 0u : kFlagNoNee);
    params.lightTreeBounces = settings.lightTreeBounces;
    params.nrcSpread = 0.0f; // the CPU renderers have no radiance cache
    params.nrcSlot = 0;
    params.nrcRecords = 0;
    return params;
}

// The integrator's hooks on the tables.
struct SceneHooks
{
    const SceneTables& tables;

    float BsdfLutLoad(uint index) const { return tables.lut[index]; }
    float4 NeuralCoatWeights4(uint index) const
    {
        const std::vector<float>& w = tables.neuralCoat;
        return index + 3 < w.size() ? float4(w[index], w[index + 1], w[index + 2], w[index + 3]) : float4(0.0f);
    }
    InstanceData LoadInstance(uint instance) const { return tables.instances[instance]; }
    uint LoadInstanceLight(uint instance) const { return tables.instanceLights[instance]; }
    uint LoadIndex(InstanceData instance, uint index) const { return tables.indices[instance.indexBuffer][index]; }
    Vertex LoadVertex(InstanceData instance, uint index) const { return tables.vertices[instance.vertexBuffer][index]; }
    MaterialData LoadMaterial(uint material) const { return tables.materials[material]; }
    TextureSetData LoadTextureSet(uint /*table*/, uint set) const { return {set * 2, set * 2 + 1, 0, 0}; }
    float4 MaterialTextureSample(uint texture, float2 uv) const
    {
        const renderer::MaterialTextures& t = *tables.textureSets[texture / 2];
        const bool color = texture % 2 == 0;
        return renderer::SampleTexture2DWrapRGBA8(color ? t.color.data() : t.surface.data(), t.width, t.height, color,
                                                  uv.x, uv.y);
    }
    LightTriangle LoadLight(uint light) const { return tables.lights[light]; }
    LightTreeNode LoadLightNode(uint node) const { return tables.lightTree[node]; }
    float3 SkyTableLoad(uint texel) const
    {
        const std::vector<float>& t = tables.sky.radiance;
        return float3(t[texel * 3], t[texel * 3 + 1], t[texel * 3 + 2]);
    }
    float4 AerialInscatterSample(SkyData s, float3 uvw) const
    {
        return renderer::SampleTexture3DClamped(tables.sky.aerialInscatter.data(), tables.sky.aerialWidth,
                                                tables.sky.aerialHeight, s.aerialSlices, uvw.x, uvw.y, uvw.z);
    }
    float4 AerialTransmittanceSample(SkyData s, float3 uvw) const
    {
        return renderer::SampleTexture3DClamped(tables.sky.aerialTransmittance.data(), tables.sky.aerialWidth,
                                                tables.sky.aerialHeight, s.aerialSlices, uvw.x, uvw.y, uvw.z);
    }
    float CloudBaseSample(CloudData c, float3 uvw) const
    {
        return renderer::SampleTexture3D(tables.clouds.base.data(), c.gridX, c.gridZ, c.gridY + 2, uvw.x, uvw.y, uvw.z);
    }
    float CloudDetailSample(CloudData /*c*/, float3 uvw) const
    {
        const u32 n = tables.clouds.detailSize;
        return renderer::SampleTexture3D(tables.clouds.detail.data(), n, n, n, uvw.x, uvw.y, uvw.z);
    }
    float CloudMajorantLoad(CloudData /*c*/, uint cell) const { return tables.clouds.majorants[cell]; }
    VolumeData LoadVolume(uint /*table*/, uint index) const { return tables.volumes[index].data; }
    float VolumeDensitySample(VolumeData v, float3 uvw) const
    {
        return renderer::SampleTexture3DClampedR16(tables.volumes[v.density].density.data(), v.sizeX + 2, v.sizeY + 2,
                                                   v.sizeZ + 2, uvw.x, uvw.y, uvw.z);
    }
    float VolumeMajorantLoad(VolumeData v, uint cell) const { return tables.volumes[v.majorants].majorants[cell]; }
    // NOLINTBEGIN(readability-identifier-naming)
#include "Cutout.hlsli"
    // NOLINTEND(readability-identifier-naming)
    HitInfo TraceClosest(float3 origin, float3 direction, uint /*bounce*/)
    {
        HitInfo hit{};
        const CutoutTest cutout = [this](u32 i, u32 p, float2 b) { return AlphaCutsOut(0, i, p, b); };
        tables.bvh->Intersect(origin, direction, 1e30f, false, hit, 0.0f, kInvalidIndex, &cutout);
        return hit;
    }
    bool TraceVisible(float3 origin, float3 direction, float tMax)
    {
        HitInfo hit{};
        const CutoutTest cutout = [this](u32 i, u32 p, float2 b) { return AlphaCutsOut(0, i, p, b); };
        return !tables.bvh->Intersect(origin, direction, tMax, true, hit, 0.0f, kInvalidIndex, &cutout);
    }
};

// --- The real-time volumes' probes (VolumeProbes.hlsl), traced by ReferenceRenderer.cpp's kernel ----

// The light a probe holds: L1 spherical harmonics of the radiance arriving there, per colour channel.
struct ProbeLight
{
    float4 red;
    float4 green;
    float4 blue;
};

// The mean over `frames` frames of the probes of `volumes` (kVolumeProbes each, in their order), as
// the GPU's running mean of them accumulates; `tables.groupMaps` tells which light groups the march
// lights the volumes with itself.
std::vector<ProbeLight> TraceVolumeProbes(const SceneTables& tables, const ReferenceSettings& settings,
                                          const SkyData& sky, std::span<const VolumeMarchData> volumes, u32 frames);

} // namespace air::pt::reference
