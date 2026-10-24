#pragma once

// Scenes: a CPU description (SceneDesc) and its GPU form (Scene / GpuScene).
//
// SceneDesc is plain data: meshes, physically based materials, instances, a camera, either a
// uniform environment or a physically based sky with a sun (Sky.h), and optionally a layer of
// clouds above it all (Clouds.h). Scene uploads it once (device-local geometry, one BLAS per mesh,
// one TLAS) and keeps the small editable parts (materials, emissive triangle list) in per-frame
// upload rings so an editor can change them between frames without stalling the GPU. Features only
// read GpuScene.
//
// Units: metres, radiance in nits (cd/m^2), camera exposure as EV100.

#include "air/renderer/Clouds.h"
#include "air/renderer/Sky.h"
#include "air/renderer/shader/SceneShared.h"
#include "air/rhi/RHIDevice.h"

#include <optional>
#include <string>
#include <vector>

namespace air::renderer
{

using float3 = hlsl::float3;

// Pinhole camera with a physical exposure. yaw = 0, pitch = 0 looks down -z; +y is up.
struct Camera
{
    float3 position = {0.0f, 1.0f, 4.0f};
    float yaw = 0.0f;   // radians, around +y
    float pitch = 0.0f; // radians, positive looks up
    float verticalFov = 0.6911f;
    // Exposure value at ISO 100 (EV100 = log2(N^2 / t) for f-number N and shutter time t seconds).
    // Bright daylight is ~15, an office ~7.
    float ev100 = 8.0f;

    float3 Forward() const;
    float3 Right() const;
    float3 Up() const;
    bool operator==(const Camera&) const = default;
};

// Scale from radiance (nits) to the exposed linear value where 1.0 is the sensor's saturation:
// 1 / (1.2 * 2^EV100) (saturation-based sensitivity, Lagarde & de Rousiers 2014).
float ExposureScale(float ev100);

struct Material
{
    std::string name;
    float3 baseColor = {0.8f, 0.8f, 0.8f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    float3 emissionColor = {1.0f, 1.0f, 1.0f}; // chromaticity; rescaled so its luminance is 1
    float emissionNits = 0.0f;                 // emitted luminance

    gpu::MaterialData ToGpu() const;
    bool operator==(const Material&) const = default;
};

struct Mesh
{
    std::string name;
    std::vector<gpu::Vertex> vertices;
    std::vector<u32> indices; // triangle list
};

struct Instance
{
    std::string name;
    u32 mesh = 0;
    u32 material = 0;
    float transform[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}; // row-major object-to-world
};

struct SceneDesc
{
    std::string name;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Instance> instances;
    Camera camera;
    float3 environment = {0.0f, 0.0f, 0.0f}; // uniform environment radiance, nits (sky.physical: unused)
    Sky sky;                                 // physical sky and sun; replaces `environment`
    Clouds clouds;                           // above every instance
};

// Mesh builders. Quads are emitted so that cross(p1 - p0, p2 - p0) points along `normal`.
Mesh MakeQuad(std::string name, float3 p0, float3 p1, float3 p2, float3 p3, float3 normal);
Mesh MakeBox(std::string name, float3 halfExtent); // centred at the origin, outward normals
Mesh MakeSphere(std::string name, float radius, u32 segments);
void SetTransform(Instance& instance, float3 translation, float yawRadians = 0.0f, float scale = 1.0f);

// Built-in scenes (all procedural and deterministic, so their images match across platforms):
//   cornell_box    the default: a 2 m box with a ceiling light, a tall box and a gold sphere
//   furnace        spheres of increasing roughness in a uniform environment: with
//                  energy-conserving white materials they become invisible
//   veach_mis      4 glossy plates reflecting 4 spherical lights of equal power and very different
//                  sizes (Veach's MIS test): NEE and BSDF sampling each fail somewhere
//   material_grid  roughness x material grid of spheres under one area light
//   sphere_field   ~2400 instanced spheres and boxes, 32 materials, physical sky and sun,
//                  emissive spheres: TLAS scale, material variety, extreme dynamic range
//   many_lights    a hall lit by ~370 lights spanning two orders of magnitude: light selection
//   door           light reaches the camera's room only through a door left ajar: indirect-
//                  dominated transport, hard for unidirectional path tracing
//   terrain        a 2.1M triangle height field with a lake, sun and sky: dense geometry
//   sunset         a plain under a low sun: the physical sky's colours, long shadows
//   cumulus        a meadow under fair-weather cumulus: the clouds' light, shadows and silver
//                  linings
SceneDesc MakeCornellBox();
SceneDesc MakeFurnaceScene();
std::vector<std::string> BuiltinSceneNames();
std::optional<SceneDesc> MakeBuiltinScene(std::string_view name);
// Checks indices, references and values. Returns the first problem or an empty string.
std::string Validate(const SceneDesc& desc);
// The highest y of any instance's vertices: the clouds' base must stay above it.
float HighestPoint(const SceneDesc& desc);

// The tables a GpuScene is made of, also used by the CPU reference renderer.
// The light list holds the triangles of emissive instances only (an instance is emissive when its
// material emits), all of an instance's triangles in order, so triangle t of instance i is entry
// FirstLightIndices()[i] + t; non-emissive instances map to gpu::kInvalidIndex. Both change with
// the materials, which is why they live in per-frame buffers rather than in the instance data.
std::vector<u32> FirstLightIndices(const SceneDesc& desc);
std::vector<gpu::LightTriangle> BuildLightTriangles(const SceneDesc& desc, double* totalPower = nullptr);
// The light tree over `lights` (gpu::LightTreeNode), which also gets each light's tree trail and
// depth. Lights without power are left out; with none, the tree is one leaf without power.
std::vector<gpu::LightTreeNode> BuildLightTree(std::vector<gpu::LightTriangle>& lights);
gpu::InstanceData MakeInstanceData(const Instance& instance, u32 vertexBuffer, u32 indexBuffer);
float3 TransformPoint(const float transform[3][4], float3 p);

struct GpuScene
{
    rhi::AccelStructHandle tlas;
    rhi::BufferHandle instances; // StructuredBuffer<gpu::InstanceData>, indexed by InstanceID()
    rhi::BufferHandle materials; // StructuredBuffer<gpu::MaterialData>
    rhi::BufferHandle lights;    // StructuredBuffer<gpu::LightTriangle>, at least one element
    rhi::BufferHandle lightTree; // StructuredBuffer<gpu::LightTreeNode> over `lights`, at least one node
    // StructuredBuffer<uint> per instance: its first entry in `lights`, gpu::kInvalidIndex if it does
    // not emit. Written together with `materials` and `lights`, so the three always agree.
    rhi::BufferHandle instanceLights;
    u32 lightCount = 0;
    float lightPower = 0.0f;    // total emitted power of lightCount triangles (watts-like, luminance * area * pi)
    gpu::SkyData sky{};         // the sun and how to read skyTable (uniform radiance when width == 0)
    rhi::BufferHandle skyTable; // StructuredBuffer<float>: 3 per texel, at least 3
    // The aerial perspective tables (RGBA16F Texture3D, SkyShared.h; none without a physical sky).
    // Their bindless indices in `sky` are left to the features.
    rhi::TextureHandle aerialInscatter;
    rhi::TextureHandle aerialTransmittance;
    // The cloud layer (enabled == 0: none, and no resources). Its bindless indices are left to the
    // features: they read the base and detail textures (R8_UNORM Texture3D) and the majorants
    // (StructuredBuffer<float>) laid out as CloudShared.h describes.
    gpu::CloudData clouds{};
    rhi::TextureHandle cloudBase;
    rhi::TextureHandle cloudDetail;
    rhi::BufferHandle cloudMajorants;
    u64 version = 0; // bumped whenever anything that changes the image changes
};

class Scene
{
public:
    // Uploads geometry, builds the acceleration structures and waits for the GPU.
    Scene(rhi::Device& device, SceneDesc desc);
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    const GpuScene& Get() const { return mGpu; }
    const SceneDesc& Desc() const { return mDesc; }

    // Editing, on the frame thread between frames. Changes reach the GPU at the next Update().
    void SetMaterial(u32 index, const Material& material);
    void SetEnvironment(float3 radiance);
    void SetSky(const Sky& sky);          // rebuilds the sky table at the next Update() (about 0.1 s)
    void SetClouds(const Clouds& clouds); // rebakes them at the next Update() (about 0.2 s)
    // Call once per frame before rendering: uploads pending edits into the next ring slot.
    void Update();

    u64 GetGeometryBytes() const { return mGeometryBytes; }
    u32 GetTriangleCount() const { return mTriangleCount; }

private:
    void Upload();
    void WriteDynamicData();

    struct GpuMesh
    {
        rhi::BufferHandle vertices;
        rhi::BufferHandle indices;
        rhi::AccelStructHandle blas;
    };

    rhi::Device& mDevice;
    SceneDesc mDesc;
    GpuScene mGpu;
    std::vector<GpuMesh> mMeshes;
    rhi::BufferHandle mInstances;
    // Host-visible rings, one slot per frame in flight + 1, so writing a slot never races the GPU.
    std::vector<rhi::BufferHandle> mMaterialRing;
    std::vector<rhi::BufferHandle> mLightRing;
    std::vector<u32> mLightRingCapacity;
    std::vector<rhi::BufferHandle> mLightTreeRing;
    std::vector<u32> mLightTreeRingCapacity;
    std::vector<rhi::BufferHandle> mInstanceLightRing;
    std::vector<rhi::BufferHandle> mSkyRing;
    std::vector<u32> mSkyRingCapacity;
    SkyTable mSkyTable; // of mSkyTableFor
    Sky mSkyTableFor;
    bool mSkyTableBuilt = false;
    // The cloud and aerial perspective resources never change once written: new clouds or a new sky
    // get new ones.
    void DestroyClouds();
    void DestroyAerial();
    Clouds mCloudsFor;
    bool mCloudsBuilt = false;
    u32 mRingSlot = 0;
    bool mDirty = true;
    u64 mGeometryBytes = 0;
    u32 mTriangleCount = 0;
};

} // namespace air::renderer
