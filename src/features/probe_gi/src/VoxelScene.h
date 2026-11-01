#pragma once

// The scene as probe GI traces it (docs/architecture.md §8.21, shaders/ProbeGiShared.h): cascades of voxels
// around the camera with a distance field, sparse surface voxels and their light, and over each cascade a
// level of the radiance cache. Owns their GPU resources; adds the passes that build a cascade (all of it when
// the scene changed, the bricks it did not cover before when the camera moved it), light the surface voxels
// and update the cache.

#include "air/rdg/RenderGraph.h"
#include "air/renderer/Scene.h"
#include "air/renderer/ShaderLoader.h"
#include "air/rhi/RHIDevice.h"

#include "ProbeGiShared.h"

#include <array>
#include <filesystem>
#include <vector>

namespace air::pgi
{

struct VoxelSettings
{
    u32 cascades = 6;             // cascades of kVoxelResolution^3 voxels, each voxel twice the last's
    float voxelSize = 0.05f;      // metres, the finest cascade's
    u32 voxelCapacity = 1u << 18; // surface voxels a cascade holds at first (doubled when they do not fit)
    u32 workCapacity = 1u << 22;  // (triangle, brick) work items of a voxelization at first (doubled likewise)
    u32 rebuildsPerFrame = 1;     // cascades rebuilt in one frame when several need it (all in the first)
    bool scroll = true;           // a cascade that moves voxelizes only the bricks it did not cover
    u32 lightingBudget = 8192;    // surface voxels of a cascade lit per frame (cycling through them all)
    float lightingBlend = 0.2f;   // weight of a new estimate in a surface voxel's light
    // Rays per face and lighting that gather a surface voxel's indirect light (radiosity); 0: the radiance
    // cache's irradiance instead (cheaper, coarse: its probes are 8 voxels apart).
    u32 radiosityRays = 1;
    // Voxels (of the face's cascade) a radiosity ray is traced before the radiance cache gives the rest (0: to
    // the end: the cache's probes converge slowly and darken the voxels' light, measured on door and many_lights).
    u32 radiosityDistance = 0;
    u32 cacheProbesPerFrame = 128; // cache probes traced per level and frame (of kCacheProbes)
    float cacheBlend = 0.15f;      // weight of a new trace in a cache probe

    bool operator==(const VoxelSettings&) const = default;
};

class VoxelScene
{
public:
    explicit VoxelScene(rhi::Device& device);
    ~VoxelScene();
    VoxelScene(const VoxelScene&) = delete;
    VoxelScene& operator=(const VoxelScene&) = delete;

    bool Initialize(const std::filesystem::path& shaderDir);

    // Moves the cascades with the camera, notices scene changes, and fills the cascades of `frame`
    // (origins, bindless indices, which are built). Call before AddPasses, every frame.
    void BeginFrame(const renderer::GpuScene& scene, const renderer::Camera& camera, const VoxelSettings& settings,
                    bool furnace, gpu::GiFrame& frame);
    // Rebuilds the cascades BeginFrame chose, lights the surface voxels and updates the radiance cache.
    // frameData / frameOffset: the GiFrame BeginFrame filled, in the frame's upload memory (ByteAddressBuffer).
    void AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, u32 frameData, u32 frameOffset,
                   u64 frameIndex);
    // Every cascade's resources, imported into `graph` (the frame's other passes read them).
    struct Imported
    {
        std::vector<rdg::TextureRef> sdf;
        std::vector<rdg::TextureRef> bricks;
        std::vector<rdg::TextureRef> voxelIndex;
        std::vector<rdg::BufferRef> voxels;
        std::vector<rdg::BufferRef> counters;
        std::vector<rdg::BufferRef> freeSlots;
        std::vector<rdg::TextureRef> cacheRadiance;
        std::vector<rdg::TextureRef> cacheDistance;
        std::vector<rdg::BufferRef> cacheIrradiance;
        std::vector<rdg::BufferRef> fresh;
    };
    const Imported& GetImported() const { return mImported; }
    // Declares reads of everything a trace through the voxels and the cache touches.
    void DeclareTraceReads(rdg::PassBuilder& builder) const;

    u32 GetCascadeCount() const { return u32(mCascades.size()); }
    // Per cascade its counters (gpu::kCounter*: work items, slots, what overflowed, the free ring's head and
    // tail), read back from the GPU (waits for it: reports only).
    std::vector<std::array<u32, gpu::kCounterBytes / 4>> ReadCounters() const;
    u32 GetBuiltCascades() const;
    u32 GetRebuilds() const { return mRebuilds; } // cascade updates so far (whole or where they moved)
    u32 GetFullRebuilds() const { return mFullRebuilds; }
    u32 GetLastRebuilt() const { return mLastRebuilt; }
    u32 GetRelights() const { return mRelights; } // frames that lit everything again because the lights changed

private:
    struct Cascade
    {
        rhi::TextureHandle sdf;
        rhi::TextureHandle bricks;
        rhi::TextureHandle voxelIndex;
        rhi::BufferHandle voxels;
        rhi::BufferHandle counters;
        rhi::BufferHandle freeSlots; // the ring of free slots of `voxels`
        rhi::TextureHandle cacheRadiance;
        rhi::TextureHandle cacheDistance;
        rhi::BufferHandle cacheIrradiance;
        rhi::BufferHandle fresh; // ring of the surface voxels new in moves (gpu::kCounterFreshHead)
        std::array<rhi::Access, 10> access{};
        std::array<int, 3> origin{};
        std::array<int, 3> builtOrigin{}; // where its voxels were built (valid when built)
        bool built = false;
        bool rebuild = false;
        bool full = true;         // the next update voxelizes everything (the scene changed, or never built)
        bool incremental = false; // this frame's update keeps the voxels of builtOrigin's box
        u32 capacity = 0;         // slots of `voxels`
        u64 checkFrom = 0;        // counters read back from updates of this frame on tell whether it fits
        bool cacheValid = false;  // its cache level has data for builtOrigin
        u64 freshUntil = 0;       // the frame (mFrames) until which it lights the voxels new where it moved
        float voxelSize = 0.0f;
    };

    void CreateResources(u32 count, u32 voxelCapacity);
    void CreatePool(Cascade& c, u32 capacity);
    void DestroyResources();
    // Grows the pools and work buffers that overflowed, from counters read back a few frames late.
    void GrowFromReadback();
    void UpdateSceneData(const renderer::GpuScene& scene);
    void AddRebuild(rdg::RenderGraph& graph, const renderer::GpuScene& scene, u32 index, u32 frameData,
                    u32 frameOffset);

    rhi::Device& mDevice;
    VoxelSettings mSettings;
    std::vector<Cascade> mCascades;
    Imported mImported;
    u32 mVoxelCapacity = 0; // the setting the pools were created with
    u32 mWorkCapacity = 0;
    bool mFurnace = false;
    // Every cascade's counters, copied each frame into a ring of kReadbackSlots readback slots.
    rhi::BufferHandle mCounterReadback;
    u32 mReadbackSlots = 0;
    u64 mFrames = 0; // frames AddPasses has run

    // Pipelines
    renderer::LazyComputePipeline mClear;
    renderer::LazyComputePipeline mBin;
    renderer::LazyComputePipeline mDistance;
    renderer::LazyComputePipeline mAllocate;
    renderer::LazyComputePipeline mAttributes;
    renderer::LazyComputePipeline mResolve;
    renderer::LazyComputePipeline mFinalize;
    renderer::LazyComputePipeline mBricks;
    renderer::LazyComputePipeline mLight;
    renderer::LazyComputePipeline mCacheReset;
    renderer::LazyComputePipeline mCacheTrace;
    renderer::LazyComputePipeline mCacheIntegrate;

    // The scene as last seen: instance boxes and triangle counts, the materials' GI textures.
    const renderer::SceneDesc* mDesc = nullptr;
    u64 mGeometryKey = 0; // instances, meshes, materials, textures: what the voxels hold
    struct InstanceBox
    {
        float lo[3];
        float hi[3];
        u32 triangles;
    };
    std::vector<InstanceBox> mInstanceBoxes;
    std::vector<rhi::TextureHandle> mGiTextures;
    rhi::BufferHandle mGiTextureTable; // StructuredBuffer<uint>

    std::vector<int> mRebuildOrder;                   // this frame's cascades to rebuild
    std::vector<std::array<int, 3>> mPreviousOrigins; // of the cascades rebuilt this frame (cache reset)
    u32 mRebuilds = 0;
    u32 mFullRebuilds = 0;
    u32 mLastRebuilt = ~0u;
    u64 mLightKey = 0; // the sun, the sky and the emitters as last seen
    bool mRelight = false;
    u32 mRelights = 0;
    u64 mFrame = 0;
};

} // namespace air::pgi
