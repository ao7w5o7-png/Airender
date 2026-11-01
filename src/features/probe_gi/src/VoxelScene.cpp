#include "VoxelScene.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>

namespace air::pgi
{

using rhi::Access;

namespace
{

constexpr int kResolution = int(gpu::kVoxelResolution);
// A cascade follows the camera closely: 24 voxels of slack left a camera moving ahead with 40 of the finest
// cascade's voxels before it instead of 64, its probes' rays went on in the coarser (darker) cascade sooner, and
// the frame was 3% darker than a still camera's there (sponza moving 3.6 cm a frame; the cost of the twice as
// many, half as large updates: 0.1 ms a frame on average, a lower peak).
constexpr int kCascadeSnap = 8;             // voxels: cascade origins are multiples of this (the cache's spacing)
constexpr int kCascadeSlack = 8;            // voxels the camera may drift from a cascade's centre before it moves
constexpr u32 kGiTextureSize = 256;         // the GI textures' largest side
constexpr u32 kGroupsMax = 16384;           // Voxelize.hlsli kGroupsMax
constexpr u32 kMaxVoxelCapacity = 1u << 21; // the pools grow up to this
constexpr u32 kMaxWorkCapacity = 1u << 25;

enum Resource : u32
{
    kSdf,
    kBricks,
    kVoxelIndex,
    kVoxels,
    kCounters,
    kFreeSlots,
    kCacheRadiance,
    kCacheDistance,
    kCacheIrradiance,
    kFresh,
};

u64 Hash(u64 h, const void* data, size_t size)
{
    const auto* bytes = static_cast<const u8*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        h = (h ^ bytes[i]) * 0x100000001b3ull;
    }
    return h;
}

float SrgbToLinear(float c)
{
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float LinearToSrgb(float c)
{
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// A texture level as linear floats: rgb and a fourth channel.
struct Level
{
    u32 width = 0;
    u32 height = 0;
    std::vector<float> texels; // 4 per texel
};

Level Halve(const Level& in)
{
    Level out;
    out.width = std::max(in.width / 2u, 1u);
    out.height = std::max(in.height / 2u, 1u);
    out.texels.resize(size_t(out.width) * out.height * 4);
    for (u32 y = 0; y < out.height; ++y)
    {
        for (u32 x = 0; x < out.width; ++x)
        {
            float sum[4] = {};
            for (u32 k = 0; k < 4; ++k)
            {
                const u32 sx = std::min(x * 2 + (k & 1u), in.width - 1);
                const u32 sy = std::min(y * 2 + (k >> 1), in.height - 1);
                for (u32 c = 0; c < 4; ++c)
                {
                    sum[c] += in.texels[(size_t(sy) * in.width + sx) * 4 + c];
                }
            }
            for (u32 c = 0; c < 4; ++c)
            {
                out.texels[(size_t(y) * out.width + x) * 4 + c] = sum[c] * 0.25f;
            }
        }
    }
    return out;
}

std::vector<u8> Pack(const Level& level)
{
    std::vector<u8> bytes(size_t(level.width) * level.height * 4);
    for (size_t i = 0; i < size_t(level.width) * level.height; ++i)
    {
        for (u32 c = 0; c < 3; ++c)
        {
            bytes[i * 4 + c] = u8(std::lround(LinearToSrgb(level.texels[i * 4 + c]) * 255.0f));
        }
        bytes[i * 4 + 3] = u8(std::lround(std::clamp(level.texels[i * 4 + 3], 0.0f, 1.0f) * 255.0f));
    }
    return bytes;
}

// A texture set's GI texture: its colour (sRGB) with its metallic, or with its alpha when a cutout uses it,
// box filtered down to at most kGiTextureSize, with every mip.
rhi::TextureHandle MakeGiTexture(rhi::Device& device, const renderer::MaterialTextures& set, bool cutout)
{
    if (set.width == 0 || set.height == 0 || set.color.size() < size_t(set.width) * set.height)
    {
        return {};
    }
    Level level;
    level.width = set.width;
    level.height = set.height;
    level.texels.resize(size_t(set.width) * set.height * 4);
    for (size_t i = 0; i < size_t(set.width) * set.height; ++i)
    {
        const u32 c = set.color[i];
        const u32 s = set.surface.size() > i ? set.surface[i] : 0u;
        level.texels[i * 4 + 0] = SrgbToLinear(float(c & 0xFFu) / 255.0f);
        level.texels[i * 4 + 1] = SrgbToLinear(float((c >> 8) & 0xFFu) / 255.0f);
        level.texels[i * 4 + 2] = SrgbToLinear(float((c >> 16) & 0xFFu) / 255.0f);
        level.texels[i * 4 + 3] = cutout ? float(c >> 24) / 255.0f : float(s >> 24) / 255.0f;
    }
    while (level.width > kGiTextureSize || level.height > kGiTextureSize)
    {
        level = Halve(level);
    }
    std::vector<std::vector<u8>> mips;
    mips.push_back(Pack(level));
    const u32 width = level.width;
    const u32 height = level.height;
    while (level.width > 1 || level.height > 1)
    {
        level = Halve(level);
        mips.push_back(Pack(level));
    }
    std::vector<std::span<const u8>> subresources;
    for (const auto& mip : mips)
    {
        subresources.emplace_back(mip.data(), mip.size());
    }
    return renderer::CreateTextureWithSubresources(device,
                                                   {.format = rhi::Format::RGBA8_SRGB,
                                                    .width = width,
                                                    .height = height,
                                                    .mipLevels = u32(mips.size()),
                                                    .usage = rhi::TextureUsage::ShaderResource,
                                                    .debugName = "ProbeGi.GiTexture"},
                                                   subresources);
}

int FloorDiv(float v, int snap)
{
    return int(std::floor(v / float(snap))) * snap;
}

} // namespace

VoxelScene::VoxelScene(rhi::Device& device) : mDevice(device) {}

VoxelScene::~VoxelScene()
{
    DestroyResources();
    for (rhi::TextureHandle t : mGiTextures)
    {
        if (t.IsValid())
        {
            mDevice.DestroyTexture(t);
        }
    }
    for (rhi::BufferHandle b : {mGiTextureTable, mCounterReadback})
    {
        if (b.IsValid())
        {
            mDevice.DestroyBuffer(b);
        }
    }
    for (renderer::LazyComputePipeline* p :
         {&mClear, &mBin, &mDistance, &mAllocate, &mAttributes, &mResolve, &mFinalize, &mBricks, &mLight, &mCacheReset,
          &mCacheTrace, &mCacheIntegrate})
    {
        p->Destroy(mDevice);
    }
}

bool VoxelScene::Initialize(const std::filesystem::path& shaderDir)
{
    const u32 voxelize = sizeof(gpu::VoxelizeConstants);
    const u32 gi = sizeof(gpu::GiConstants);
    mClear.Load(mDevice, shaderDir, "VoxelizeClear.cs", "ProbeGi.Voxelize.Clear", voxelize);
    mBin.Load(mDevice, shaderDir, "VoxelizeBin.cs", "ProbeGi.Voxelize.Bin", voxelize);
    mDistance.Load(mDevice, shaderDir, "VoxelizeDistance.cs", "ProbeGi.Voxelize.Distance", voxelize);
    mAllocate.Load(mDevice, shaderDir, "VoxelizeAllocate.cs", "ProbeGi.Voxelize.Allocate", voxelize);
    mAttributes.Load(mDevice, shaderDir, "VoxelizeAttributes.cs", "ProbeGi.Voxelize.Attributes", voxelize);
    mResolve.Load(mDevice, shaderDir, "VoxelizeResolve.cs", "ProbeGi.Voxelize.Resolve", voxelize);
    mFinalize.Load(mDevice, shaderDir, "VoxelizeFinalize.cs", "ProbeGi.Voxelize.Finalize", voxelize);
    mBricks.Load(mDevice, shaderDir, "VoxelizeBricks.cs", "ProbeGi.Voxelize.Bricks", voxelize);
    mLight.Load(mDevice, shaderDir, "VoxelLight.cs", "ProbeGi.VoxelLight", gi);
    mCacheReset.Load(mDevice, shaderDir, "CacheReset.cs", "ProbeGi.Cache.Reset", gi);
    mCacheTrace.Load(mDevice, shaderDir, "CacheTrace.cs", "ProbeGi.Cache.Trace", gi);
    mCacheIntegrate.Load(mDevice, shaderDir, "CacheIntegrate.cs", "ProbeGi.Cache.Integrate", gi);
    bool ok = true;
    for (const renderer::LazyComputePipeline* p :
         {&mClear, &mBin, &mDistance, &mAllocate, &mAttributes, &mResolve, &mFinalize, &mBricks, &mLight, &mCacheReset,
          &mCacheTrace, &mCacheIntegrate})
    {
        ok = ok && p->IsAvailable();
    }
    if (!ok)
    {
        AIR_LOG(Error, "ProbeGi", "voxel scene shaders are missing from {}", shaderDir.string());
    }
    return ok;
}

void VoxelScene::CreateResources(u32 count, u32 voxelCapacity)
{
    DestroyResources();
    mCascades.resize(count);
    mVoxelCapacity = voxelCapacity;
    const u32 n = gpu::kVoxelResolution;
    const u32 atlasWidth = gpu::kCacheResolution * gpu::kCacheResolution * gpu::kCacheTexels;
    const u32 atlasHeight = gpu::kCacheResolution * gpu::kCacheTexels;
    for (Cascade& c : mCascades)
    {
        const auto uav = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        c.sdf = mDevice.CreateTexture({.dimension = rhi::TextureDimension::Tex3D,
                                       .format = rhi::Format::R8_UNORM,
                                       .width = n,
                                       .height = n,
                                       .depthOrArraySize = n,
                                       .usage = uav,
                                       .debugName = "ProbeGi.Sdf"});
        c.bricks = mDevice.CreateTexture({.dimension = rhi::TextureDimension::Tex3D,
                                          .format = rhi::Format::R32_UINT,
                                          .width = gpu::kBrickResolution,
                                          .height = gpu::kBrickResolution,
                                          .depthOrArraySize = gpu::kBrickResolution,
                                          .usage = uav,
                                          .debugName = "ProbeGi.Bricks"});
        c.voxelIndex = mDevice.CreateTexture({.dimension = rhi::TextureDimension::Tex3D,
                                              .format = rhi::Format::R32_UINT,
                                              .width = n,
                                              .height = n,
                                              .depthOrArraySize = n,
                                              .usage = uav,
                                              .debugName = "ProbeGi.VoxelIndex"});
        c.counters = mDevice.CreateBuffer(
            {.size = gpu::kCounterBytes,
             .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource | rhi::BufferUsage::CopySrc,
             .debugName = "ProbeGi.VoxelCounters"});
        c.fresh = mDevice.CreateBuffer({.size = u64(gpu::kFreshCapacity) * 8,
                                        .stride = 8,
                                        .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                        .debugName = "ProbeGi.FreshVoxels"});
        c.cacheRadiance = mDevice.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                                 .width = atlasWidth,
                                                 .height = atlasHeight,
                                                 .usage = uav,
                                                 .debugName = "ProbeGi.CacheRadiance"});
        c.cacheDistance = mDevice.CreateTexture({.format = rhi::Format::RG16_FLOAT,
                                                 .width = atlasWidth,
                                                 .height = atlasHeight,
                                                 .usage = uav,
                                                 .debugName = "ProbeGi.CacheDistance"});
        c.cacheIrradiance =
            mDevice.CreateBuffer({.size = u64(gpu::kCacheProbes) * sizeof(gpu::ProbeSh),
                                  .stride = sizeof(gpu::ProbeSh),
                                  .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                  .debugName = "ProbeGi.CacheIrradiance"});
        c.access.fill(Access::None);
        CreatePool(c, voxelCapacity);
        c.built = false;
        c.full = true;
        c.cacheValid = false;
    }
    if (!mCounterReadback.IsValid())
    {
        // A frame's copy is read once the frames in flight after it began: its GPU work is done by then.
        mReadbackSlots = mDevice.GetFramesInFlight() + 2;
        mCounterReadback = mDevice.CreateBuffer({.size = u64(mReadbackSlots) * gpu::kMaxCascades * gpu::kCounterBytes,
                                                 .usage = rhi::BufferUsage::CopyDst,
                                                 .memory = rhi::MemoryLocation::Readback,
                                                 .debugName = "ProbeGi.CounterReadback"});
    }
}

void VoxelScene::CreatePool(Cascade& c, u32 capacity)
{
    for (rhi::BufferHandle b : {c.voxels, c.freeSlots})
    {
        if (b.IsValid())
        {
            mDevice.DestroyBuffer(b);
        }
    }
    c.capacity = capacity;
    c.voxels = mDevice.CreateBuffer({.size = u64(capacity) * sizeof(gpu::SurfaceVoxel),
                                     .stride = sizeof(gpu::SurfaceVoxel),
                                     .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                     .debugName = "ProbeGi.SurfaceVoxels"});
    c.freeSlots = mDevice.CreateBuffer({.size = u64(capacity) * sizeof(u32),
                                        .stride = sizeof(u32),
                                        .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                                        .debugName = "ProbeGi.FreeSlots"});
    c.access[kVoxels] = Access::None;
    c.access[kFreeSlots] = Access::None;
    c.checkFrom = ~0ull;
}

void VoxelScene::GrowFromReadback()
{
    if (!mCounterReadback.IsValid() || mFrames + 1 < mReadbackSlots)
    {
        return;
    }
    const auto* mapped = static_cast<const u8*>(mDevice.GetMappedPointer(mCounterReadback));
    if (!mapped)
    {
        return;
    }
    // The copy of kReadbackSlots - 1 frames ago.
    const u64 frame = mFrames + 1 - mReadbackSlots;
    bool waited = false;
    for (u32 i = 0; i < u32(mCascades.size()); ++i)
    {
        Cascade& c = mCascades[i];
        if (frame < c.checkFrom)
        {
            continue; // not updated since it was created or grew: the counters are not its
        }
        u32 counters[gpu::kCounterBytes / 4];
        std::memcpy(counters, mapped + (size_t(frame % mReadbackSlots) * gpu::kMaxCascades + i) * gpu::kCounterBytes,
                    sizeof(counters));
        const u32 lostVoxels = counters[gpu::kCounterVoxelOverflow / 4];
        const u32 lostWork = counters[gpu::kCounterWorkOverflow / 4];
        const bool voxels = lostVoxels > 0 && c.capacity < kMaxVoxelCapacity;
        const bool work = lostWork > 0 && mWorkCapacity < kMaxWorkCapacity;
        if (!voxels && !work)
        {
            continue;
        }
        if (voxels)
        {
            if (!waited)
            {
                mDevice.WaitIdle(); // frames in flight may still read the pool
                waited = true;
            }
            CreatePool(c, std::min(c.capacity * 2, kMaxVoxelCapacity));
            c.built = false;
            AIR_LOG(Info, "ProbeGi", "cascade {}: {} surface voxels did not fit, {} slots now", i, lostVoxels,
                    c.capacity);
        }
        if (work)
        {
            mWorkCapacity = std::min(mWorkCapacity * 2, kMaxWorkCapacity);
            AIR_LOG(Info, "ProbeGi", "cascade {}: {} voxelization work items did not fit, {} now", i, lostWork,
                    mWorkCapacity);
        }
        c.full = true;
        c.rebuild = true;
        c.checkFrom = ~0ull;
    }
}

void VoxelScene::DestroyResources()
{
    if (mCascades.empty())
    {
        return;
    }
    for (Cascade& c : mCascades)
    {
        for (rhi::TextureHandle t : {c.sdf, c.bricks, c.voxelIndex, c.cacheRadiance, c.cacheDistance})
        {
            if (t.IsValid())
            {
                mDevice.DestroyTexture(t);
            }
        }
        for (rhi::BufferHandle b : {c.voxels, c.counters, c.freeSlots, c.cacheIrradiance, c.fresh})
        {
            if (b.IsValid())
            {
                mDevice.DestroyBuffer(b);
            }
        }
    }
    mCascades.clear();
}

std::vector<std::array<u32, gpu::kCounterBytes / 4>> VoxelScene::ReadCounters() const
{
    std::vector<std::array<u32, gpu::kCounterBytes / 4>> result;
    for (const Cascade& c : mCascades)
    {
        std::array<u32, gpu::kCounterBytes / 4> values{};
        if (c.built)
        {
            const std::vector<u8> bytes = renderer::ReadbackBuffer(mDevice, c.counters, c.access[kCounters]);
            if (bytes.size() >= sizeof(values))
            {
                std::memcpy(values.data(), bytes.data(), sizeof(values));
            }
        }
        result.push_back(values);
    }
    return result;
}

u32 VoxelScene::GetBuiltCascades() const
{
    return u32(std::count_if(mCascades.begin(), mCascades.end(), [](const Cascade& c) { return c.built; }));
}

void VoxelScene::UpdateSceneData(const renderer::GpuScene& scene)
{
    const renderer::SceneDesc* desc = scene.desc;
    if (!desc)
    {
        return;
    }
    // What the voxels hold: the instances (their meshes and places), the materials and the texture sets.
    u64 key = 0xcbf29ce484222325ull;
    key = Hash(key, &desc, sizeof(desc));
    for (const renderer::Instance& instance : desc->instances)
    {
        key = Hash(key, &instance.mesh, sizeof(instance.mesh));
        key = Hash(key, &instance.material, sizeof(instance.material));
        key = Hash(key, instance.transform, sizeof(instance.transform));
    }
    for (const renderer::Material& m : desc->materials)
    {
        key = Hash(key, &m.baseColor, sizeof(m.baseColor));
        key = Hash(key, &m.roughness, sizeof(m.roughness));
        key = Hash(key, &m.metallic, sizeof(m.metallic));
        key = Hash(key, &m.textures, sizeof(m.textures));
        key = Hash(key, &m.uvScale, sizeof(m.uvScale));
        key = Hash(key, &m.alphaCutoff, sizeof(m.alphaCutoff));
    }
    const size_t meshCount = desc->meshes.size();
    key = Hash(key, &meshCount, sizeof(meshCount));
    const size_t setCount = desc->textureSets.size();
    key = Hash(key, &setCount, sizeof(setCount));
    const bool newScene = desc != mDesc;
    if (key == mGeometryKey && !newScene)
    {
        return;
    }
    mGeometryKey = key;
    mDesc = desc;
    // Instance boxes and triangle counts.
    mInstanceBoxes.resize(desc->instances.size());
    for (size_t i = 0; i < desc->instances.size(); ++i)
    {
        const renderer::Instance& instance = desc->instances[i];
        const renderer::Mesh& mesh = desc->meshes[instance.mesh];
        InstanceBox box{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}, u32(mesh.indices.size() / 3)};
        for (const air::gpu::Vertex& v : mesh.vertices)
        {
            const hlsl::float3 p = renderer::TransformPoint(instance.transform, v.position);
            const float c[3] = {p.x, p.y, p.z};
            for (int a = 0; a < 3; ++a)
            {
                box.lo[a] = std::min(box.lo[a], c[a]);
                box.hi[a] = std::max(box.hi[a], c[a]);
            }
        }
        mInstanceBoxes[i] = box;
    }
    // The materials' GI textures, rebuilt with the scene's texture sets.
    for (rhi::TextureHandle t : mGiTextures)
    {
        if (t.IsValid())
        {
            mDevice.DestroyTexture(t);
        }
    }
    mGiTextures.assign(desc->textureSets.size(), {});
    std::vector<u32> table(std::max<size_t>(desc->textureSets.size(), 1), air::gpu::kInvalidIndex);
    for (size_t i = 0; i < desc->textureSets.size(); ++i)
    {
        bool cutout = false;
        for (const renderer::Material& m : desc->materials)
        {
            cutout = cutout || (m.textures == i && m.alphaCutoff > 0.0f);
        }
        mGiTextures[i] = MakeGiTexture(mDevice, desc->textureSets[i], cutout);
        if (mGiTextures[i].IsValid())
        {
            table[i] = mDevice.GetBindlessSRV(mGiTextures[i]);
        }
    }
    if (mGiTextureTable.IsValid())
    {
        mDevice.DestroyBuffer(mGiTextureTable);
    }
    mGiTextureTable = renderer::CreateBufferWithData(
        mDevice,
        {.stride = sizeof(u32), .usage = rhi::BufferUsage::ShaderResource, .debugName = "ProbeGi.GiTextureTable"},
        table.data(), table.size() * sizeof(u32));
    // Everything is voxelized again; a new scene's old voxels are of no use meanwhile.
    for (Cascade& c : mCascades)
    {
        c.rebuild = true;
        c.full = true;
        if (newScene)
        {
            c.built = false;
            c.cacheValid = false;
        }
    }
    AIR_LOG(Info, "ProbeGi", "scene changed: {} instances, {} texture sets", desc->instances.size(),
            desc->textureSets.size());
}

void VoxelScene::BeginFrame(const renderer::GpuScene& scene, const renderer::Camera& camera,
                            const VoxelSettings& settings, bool furnace, gpu::GiFrame& frame)
{
    const u32 count = std::clamp(settings.cascades, 1u, gpu::kMaxCascades);
    if (mCascades.size() != count || settings.voxelSize != mSettings.voxelSize ||
        settings.voxelCapacity != mVoxelCapacity)
    {
        mDevice.WaitIdle(); // frames in flight may still read the cascades
        CreateResources(count, std::max(settings.voxelCapacity, 1024u));
        mGeometryKey = 0;
    }
    if (mWorkCapacity == 0 || settings.workCapacity != mSettings.workCapacity)
    {
        mWorkCapacity = std::max(settings.workCapacity, 1024u);
    }
    mSettings = settings;
    if (furnace != mFurnace)
    {
        mFurnace = furnace;
        mGeometryKey = 0; // the furnace voxels are white
    }
    UpdateSceneData(scene);
    GrowFromReadback();
    // The lights: a change of the sun, the sky or the emitters lights every voxel and cache probe again at once (their
    // light would otherwise follow over many frames, at the blends' rates).
    u64 lightKey = Hash(1469598103934665603ull, &scene.sky, sizeof(scene.sky));
    lightKey = Hash(lightKey, &scene.lightCount, sizeof(scene.lightCount));
    lightKey = Hash(lightKey, &scene.lightPower, sizeof(scene.lightPower));
    mRelight = mLightKey != 0 && lightKey != mLightKey;
    mRelights += mRelight ? 1u : 0u;
    mLightKey = lightKey;

    const float position[3] = {camera.position.x, camera.position.y, camera.position.z};
    for (u32 i = 0; i < count; ++i)
    {
        Cascade& c = mCascades[i];
        c.voxelSize = settings.voxelSize * float(1u << i);
        std::array<int, 3> desired{};
        bool far = false;
        for (int a = 0; a < 3; ++a)
        {
            const float v = position[a] / c.voxelSize;
            desired[a] = FloorDiv(v - float(kResolution / 2), kCascadeSnap);
            far = far || std::abs(v - float(c.builtOrigin[a] + kResolution / 2)) > float(kCascadeSlack);
        }
        if (!c.built || far)
        {
            c.rebuild = true;
        }
        c.origin = c.rebuild ? desired : c.builtOrigin;
        // A move shorter than the cascade keeps the voxels the old and the new box share.
        c.incremental = settings.scroll && c.built && !c.full;
        for (int a = 0; a < 3; ++a)
        {
            c.incremental = c.incremental && std::abs(c.origin[a] - c.builtOrigin[a]) < kResolution;
        }
    }
    // Which cascades to rebuild now: the finest first, all at once while none is built.
    mRebuildOrder.clear();
    const bool none = GetBuiltCascades() == 0;
    for (u32 i = 0; i < count; ++i)
    {
        if (mCascades[i].rebuild && (none || mRebuildOrder.size() < std::max(settings.rebuildsPerFrame, 1u)))
        {
            mRebuildOrder.push_back(int(i));
        }
    }

    for (u32 i = 0; i < gpu::kMaxCascades; ++i)
    {
        gpu::CascadeData& d = frame.cascades[i];
        d = {};
        if (i >= count)
        {
            continue;
        }
        const Cascade& c = mCascades[i];
        const bool now = std::find(mRebuildOrder.begin(), mRebuildOrder.end(), int(i)) != mRebuildOrder.end();
        const std::array<int, 3>& origin = now ? c.origin : c.builtOrigin;
        d.originX = origin[0];
        d.originY = origin[1];
        d.originZ = origin[2];
        d.voxelSize = c.voxelSize;
        d.sdf = mDevice.GetBindlessSRV(c.sdf);
        d.bricks = mDevice.GetBindlessSRV(c.bricks);
        d.voxelIndex = mDevice.GetBindlessSRV(c.voxelIndex);
        d.voxels = mDevice.GetBindlessSRV(c.voxels);
        d.valid = (c.built || now) ? 1u : 0u;
        d.cacheRadiance = mDevice.GetBindlessSRV(c.cacheRadiance);
        d.cacheDistance = mDevice.GetBindlessSRV(c.cacheDistance);
        d.cacheIrradiance = mDevice.GetBindlessSRV(c.cacheIrradiance);
        // What this frame's updates do: a cascade built whole now is lit and traced whole, the others (and one
        // that moved: its new voxels start from the coarser cascade's light) a budget.
        const bool whole = (now && !c.incremental) || (mRelight && c.built);
        d.voxelsWritable = mDevice.GetBindlessUAV(c.voxels);
        d.counters = mDevice.GetBindlessSRV(c.counters);
        d.lightBudget = whole ? 0u : std::max(settings.lightingBudget, 64u);
        d.lightBlend = whole ? 1.0f : settings.lightingBlend;
        d.cacheProbes = whole ? 0u : std::clamp(settings.cacheProbesPerFrame, 1u, gpu::kCacheProbes);
        d.cacheRadianceWritable = mDevice.GetBindlessUAV(c.cacheRadiance);
        d.cacheDistanceWritable = mDevice.GetBindlessUAV(c.cacheDistance);
        d.cacheIrradianceWritable = mDevice.GetBindlessUAV(c.cacheIrradiance);
    }
    frame.cascadeCount = count;
}

void VoxelScene::DeclareTraceReads(rdg::PassBuilder& builder) const
{
    for (size_t i = 0; i < mImported.sdf.size(); ++i)
    {
        builder.Read(mImported.sdf[i], Access::SRVCompute);
        builder.Read(mImported.bricks[i], Access::SRVCompute);
        builder.Read(mImported.voxelIndex[i], Access::SRVCompute);
        builder.Read(mImported.voxels[i], Access::SRVCompute);
        builder.Read(mImported.cacheRadiance[i], Access::SRVCompute);
        builder.Read(mImported.cacheDistance[i], Access::SRVCompute);
        builder.Read(mImported.cacheIrradiance[i], Access::SRVCompute);
    }
}

void VoxelScene::AddRebuild(rdg::RenderGraph& graph, const renderer::GpuScene& scene, u32 index, u32 frameData,
                            u32 frameOffset)
{
    Cascade& c = mCascades[index];
    // The instances whose boxes reach the cascade's box and its band, but (when it moved) not those that lie
    // so deep inside its old box that their band reaches none of the voxels it did not cover.
    const float margin = (gpu::kVoxelBand + 1.0f) * c.voxelSize;
    float lo[3];
    float hi[3];
    float keepLo[3];
    float keepHi[3];
    for (int a = 0; a < 3; ++a)
    {
        lo[a] = float(c.origin[a]) * c.voxelSize - margin;
        hi[a] = float(c.origin[a] + kResolution) * c.voxelSize + margin;
        keepLo[a] = float(c.builtOrigin[a]) * c.voxelSize + margin;
        keepHi[a] = float(c.builtOrigin[a] + kResolution) * c.voxelSize - margin;
    }
    std::vector<gpu::VoxelizeRange> ranges;
    u32 triangles = 0;
    for (size_t i = 0; i < mInstanceBoxes.size(); ++i)
    {
        const InstanceBox& b = mInstanceBoxes[i];
        bool overlap = b.triangles > 0;
        bool kept = c.incremental;
        for (int a = 0; a < 3; ++a)
        {
            overlap = overlap && b.hi[a] >= lo[a] && b.lo[a] <= hi[a];
            kept = kept && b.lo[a] >= keepLo[a] && b.hi[a] <= keepHi[a];
        }
        if (overlap && !kept)
        {
            ranges.push_back({u32(i), triangles, b.triangles, 0});
            triangles += b.triangles;
        }
    }
    // The frame's upload memory holds them until the GPU has finished it.
    const rdg::UploadAllocator::Allocation rangeData =
        graph.AllocateUpload(std::max<size_t>(ranges.size(), 1) * sizeof(gpu::VoxelizeRange), 16);
    if (!ranges.empty())
    {
        std::memcpy(rangeData.cpu, ranges.data(), ranges.size() * sizeof(gpu::VoxelizeRange));
    }

    const u32 n = gpu::kVoxelResolution;
    const auto uav = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
    rdg::TextureRef distances = graph.CreateTexture({.dimension = rhi::TextureDimension::Tex3D,
                                                     .format = rhi::Format::R32_UINT,
                                                     .width = n,
                                                     .height = n,
                                                     .depthOrArraySize = n,
                                                     .usage = uav,
                                                     .debugName = "ProbeGi.Voxelize.Distances"});
    auto brickTexture = [&](const char* name)
    {
        return graph.CreateTexture({.dimension = rhi::TextureDimension::Tex3D,
                                    .format = rhi::Format::R32_UINT,
                                    .width = gpu::kBrickResolution,
                                    .height = gpu::kBrickResolution,
                                    .depthOrArraySize = gpu::kBrickResolution,
                                    .usage = uav,
                                    .debugName = name});
    };
    rdg::TextureRef occupancy = brickTexture("ProbeGi.Voxelize.Occupancy");
    rdg::TextureRef bricksX = brickTexture("ProbeGi.Voxelize.BricksX");
    rdg::TextureRef bricksY = brickTexture("ProbeGi.Voxelize.BricksY");
    const u32 workCapacity = mWorkCapacity;
    rdg::BufferRef workItems =
        graph.CreateBuffer({.size = u64(workCapacity) * 8,
                            .stride = 8,
                            .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                            .debugName = "ProbeGi.Voxelize.WorkItems"});
    rdg::BufferRef accumulators =
        graph.CreateBuffer({.size = u64(c.capacity) * gpu::kAccumulatorStride * sizeof(u32),
                            .stride = sizeof(u32),
                            .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                            .debugName = "ProbeGi.Voxelize.Accumulators"});
    const rdg::TextureRef sdf = mImported.sdf[index];
    const rdg::TextureRef bricks = mImported.bricks[index];
    const rdg::TextureRef voxelIndex = mImported.voxelIndex[index];
    const rdg::BufferRef voxels = mImported.voxels[index];
    const rdg::BufferRef counters = mImported.counters[index];
    const rdg::BufferRef freeSlots = mImported.freeSlots[index];
    const rdg::BufferRef fresh = mImported.fresh[index];

    gpu::VoxelizeConstants base{};
    base.frameData = frameData;
    base.frameOffset = frameOffset;
    base.cascade = index;
    base.instances = mDevice.GetBindlessSRV(scene.instances);
    base.materials = mDevice.GetBindlessSRV(scene.materials);
    base.giTextures = mDevice.GetBindlessSRV(mGiTextureTable);
    base.ranges = rangeData.srv;
    base.rangeCount = u32(ranges.size());
    base.triangleCount = triangles;
    base.workCapacity = workCapacity;
    base.voxelCapacity = c.capacity;
    base.furnace = mFurnace ? 1u : 0u;
    base.rangeOffset = u32(rangeData.offset);
    base.keepX = c.builtOrigin[0] - c.origin[0];
    base.keepY = c.builtOrigin[1] - c.origin[1];
    base.keepZ = c.builtOrigin[2] - c.origin[2];
    base.incremental = c.incremental ? 1u : 0u;

    const std::string prefix = "ProbeGi.Voxelize[" + std::to_string(index) + "].";
    const rhi::PipelineHandle clear = mClear.Get(mDevice);
    graph.AddPass(
        prefix + "Clear", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Write(distances, Access::UAVCompute);
            b.Write(voxelIndex, Access::UAVCompute);
            b.Write(occupancy, Access::UAVCompute);
            b.Write(counters, Access::UAVCompute);
            b.Write(voxels, Access::UAVCompute);
            b.Write(freeSlots, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            base.distances = ctx.UAV(distances);
            base.voxelIndex = ctx.UAV(voxelIndex);
            base.occupancy = ctx.UAV(occupancy);
            base.counters = ctx.UAV(counters);
            base.voxels = ctx.UAV(voxels);
            base.freeSlots = ctx.UAV(freeSlots);
            ctx.cmd.SetPipeline(clear);
            ctx.cmd.PushConstants(&base, sizeof(base));
            ctx.cmd.Dispatch(n / 4, n / 4, n / 4);
        });
    if (!ranges.empty())
    {
        const rhi::PipelineHandle bin = mBin.Get(mDevice);
        const u32 binGroups = std::min((triangles + 63) / 64, kGroupsMax);
        graph.AddPass(
            prefix + "Bin", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Write(counters, Access::UAVCompute);
                b.Write(workItems, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.counters = ctx.UAV(counters);
                base.workItems = ctx.UAV(workItems);
                ctx.cmd.SetPipeline(bin);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(binGroups, 1, 1);
            });
        const rhi::PipelineHandle distance = mDistance.Get(mDevice);
        graph.AddPass(
            prefix + "Distance", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(counters, Access::SRVCompute);
                b.Read(workItems, Access::SRVCompute);
                b.Write(distances, Access::UAVCompute);
                b.Write(voxelIndex, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.counters = ctx.SRV(counters);
                base.workItems = ctx.SRV(workItems);
                base.distances = ctx.UAV(distances);
                base.voxelIndex = ctx.UAV(voxelIndex);
                ctx.cmd.SetPipeline(distance);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(kGroupsMax, 1, 1);
            });
    }
    const rhi::PipelineHandle allocate = mAllocate.Get(mDevice);
    graph.AddPass(
        prefix + "Allocate", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Write(voxelIndex, Access::UAVCompute);
            b.Write(counters, Access::UAVCompute);
            b.Write(voxels, Access::UAVCompute);
            b.Write(accumulators, Access::UAVCompute);
            b.Write(freeSlots, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            base.voxelIndex = ctx.UAV(voxelIndex);
            base.counters = ctx.UAV(counters);
            base.voxels = ctx.UAV(voxels);
            base.accumulators = ctx.UAV(accumulators);
            base.freeSlots = ctx.UAV(freeSlots);
            ctx.cmd.SetPipeline(allocate);
            ctx.cmd.PushConstants(&base, sizeof(base));
            ctx.cmd.Dispatch(n / 4, n / 4, n / 4);
        });
    if (!ranges.empty())
    {
        const rhi::PipelineHandle attributes = mAttributes.Get(mDevice);
        graph.AddPass(
            prefix + "Attributes", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(counters, Access::SRVCompute);
                b.Read(workItems, Access::SRVCompute);
                b.Write(voxelIndex, Access::UAVCompute);
                b.Write(accumulators, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.counters = ctx.SRV(counters);
                base.workItems = ctx.SRV(workItems);
                base.voxelIndex = ctx.UAV(voxelIndex);
                base.accumulators = ctx.UAV(accumulators);
                ctx.cmd.SetPipeline(attributes);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(kGroupsMax, 1, 1);
            });
    }
    // Resolve: a moved cascade's new voxels start from the light of the next coarser cascade's.
    const rhi::PipelineHandle resolve = mResolve.Get(mDevice);
    const bool coarser = index + 1 < u32(mCascades.size());
    graph.AddPass(
        prefix + "Resolve", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Write(counters, Access::UAVCompute);
            b.Read(voxelIndex, Access::SRVCompute);
            b.Write(accumulators, Access::UAVCompute);
            b.Write(voxels, Access::UAVCompute);
            b.Write(fresh, Access::UAVCompute);
            if (coarser)
            {
                b.Read(mImported.voxelIndex[index + 1], Access::SRVCompute);
                b.Read(mImported.voxels[index + 1], Access::SRVCompute);
            }
        },
        [=](rdg::PassContext& ctx) mutable
        {
            base.counters = ctx.UAV(counters);
            base.voxelIndex = ctx.SRV(voxelIndex);
            base.accumulators = ctx.UAV(accumulators);
            base.voxels = ctx.UAV(voxels);
            base.fresh = ctx.UAV(fresh);
            ctx.cmd.SetPipeline(resolve);
            ctx.cmd.PushConstants(&base, sizeof(base));
            ctx.cmd.Dispatch(n / 4, n / 4, n / 4);
        });
    const rhi::PipelineHandle finalize = mFinalize.Get(mDevice);
    graph.AddPass(
        prefix + "Finalize", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Write(distances, Access::UAVCompute);
            b.Write(sdf, Access::UAVCompute);
            b.Write(voxelIndex, Access::UAVCompute);
            b.Write(occupancy, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            base.distances = ctx.UAV(distances);
            base.sdf = ctx.UAV(sdf);
            base.voxelIndex = ctx.UAV(voxelIndex);
            base.occupancy = ctx.UAV(occupancy);
            ctx.cmd.SetPipeline(finalize);
            ctx.cmd.PushConstants(&base, sizeof(base));
            ctx.cmd.Dispatch(n / 4, n / 4, n / 4);
        });
    const rhi::PipelineHandle bricksPipeline = mBricks.Get(mDevice);
    const rdg::TextureRef inputs[3] = {occupancy, bricksX, bricksY};
    const rdg::TextureRef outputs[3] = {bricksX, bricksY, bricks};
    for (u32 axis = 0; axis < 3; ++axis)
    {
        const rdg::TextureRef in = inputs[axis];
        const rdg::TextureRef out = outputs[axis];
        graph.AddPass(
            prefix + "Bricks" + char('X' + axis), rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(in, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                base.axis = axis;
                base.bricksIn = ctx.SRV(in);
                base.bricks = ctx.UAV(out);
                ctx.cmd.SetPipeline(bricksPipeline);
                ctx.cmd.PushConstants(&base, sizeof(base));
                ctx.cmd.Dispatch(gpu::kBrickResolution / 4, gpu::kBrickResolution / 4, gpu::kBrickResolution / 4);
            });
    }
}

void VoxelScene::AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene, u32 frameData, u32 frameOffset,
                           u64 frameIndex)
{
    mFrame = frameIndex;
    const u32 count = u32(mCascades.size());
    // Import every cascade; they end where traces read them.
    mImported = {};
    for (Cascade& c : mCascades)
    {
        auto importTexture = [&](rhi::TextureHandle t, Resource r)
        {
            const rdg::TextureRef ref = graph.ImportTexture(t, c.access[r], Access::SRVCompute);
            c.access[r] = Access::SRVCompute;
            return ref;
        };
        auto importBuffer = [&](rhi::BufferHandle b, Resource r)
        {
            const rdg::BufferRef ref = graph.ImportBuffer(b, c.access[r], Access::SRVCompute);
            c.access[r] = Access::SRVCompute;
            return ref;
        };
        mImported.sdf.push_back(importTexture(c.sdf, kSdf));
        mImported.bricks.push_back(importTexture(c.bricks, kBricks));
        mImported.voxelIndex.push_back(importTexture(c.voxelIndex, kVoxelIndex));
        mImported.voxels.push_back(importBuffer(c.voxels, kVoxels));
        mImported.counters.push_back(importBuffer(c.counters, kCounters));
        mImported.freeSlots.push_back(importBuffer(c.freeSlots, kFreeSlots));
        mImported.cacheRadiance.push_back(importTexture(c.cacheRadiance, kCacheRadiance));
        mImported.cacheDistance.push_back(importTexture(c.cacheDistance, kCacheDistance));
        mImported.cacheIrradiance.push_back(importBuffer(c.cacheIrradiance, kCacheIrradiance));
        mImported.fresh.push_back(importBuffer(c.fresh, kFresh));
    }

    // Rebuilds; a moved cascade's cache keeps the probes it still covers.
    std::vector<bool> rebuilt(count, false); // whole
    const rhi::PipelineHandle cacheReset = mCacheReset.Get(mDevice);
    for (int i : mRebuildOrder)
    {
        Cascade& c = mCascades[u32(i)];
        AddRebuild(graph, scene, u32(i), frameData, frameOffset);
        gpu::GiConstants reset{};
        reset.frameData = frameData;
        reset.frameOffset = frameOffset;
        reset.cascade = u32(i);
        reset.input0 = u32(c.builtOrigin[0] / 8);
        reset.input1 = u32(c.builtOrigin[1] / 8);
        reset.input2 = u32(c.builtOrigin[2] / 8);
        reset.input3 = c.cacheValid ? 0u : 1u;
        const rdg::TextureRef radiance = mImported.cacheRadiance[u32(i)];
        const rdg::TextureRef moments = mImported.cacheDistance[u32(i)];
        const rdg::BufferRef irradiance = mImported.cacheIrradiance[u32(i)];
        graph.AddPass(
            "ProbeGi.Cache[" + std::to_string(i) + "].Reset", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Write(radiance, Access::UAVCompute);
                b.Write(moments, Access::UAVCompute);
                b.Write(irradiance, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                reset.input4 = ctx.UAV(radiance);
                reset.input5 = ctx.UAV(moments);
                reset.input6 = ctx.UAV(irradiance);
                ctx.cmd.SetPipeline(cacheReset);
                ctx.cmd.PushConstants(&reset, sizeof(reset));
                ctx.cmd.Dispatch(gpu::kCacheProbes / 64, 1, 1);
            });
        rebuilt[u32(i)] = !c.incremental;
        if (c.incremental)
        {
            c.freshUntil = mFrames + gpu::kFreshFrames + 1; // its new voxels are lit while they are young
        }
        mFullRebuilds += c.incremental ? 0u : 1u;
        c.builtOrigin = c.origin;
        c.built = true;
        c.rebuild = false;
        c.full = false;
        c.incremental = false;
        c.cacheValid = true;
        c.checkFrom = std::min(c.checkFrom, mFrames);
        ++mRebuilds;
        mLastRebuilt = u32(i);
    }

    // Every cascade's counters into this frame's readback slot (GrowFromReadback reads them).
    if (count > 0)
    {
        const u32 slot = u32(mFrames % mReadbackSlots);
        const rdg::BufferRef readback = graph.ImportBuffer(mCounterReadback, Access::CopyDst, Access::CopyDst);
        const std::vector<rdg::BufferRef> counters = mImported.counters;
        graph.AddPass(
            "ProbeGi.Counters", rdg::PassFlags::Copy | rdg::PassFlags::NeverCull,
            [&](rdg::PassBuilder& b)
            {
                for (const rdg::BufferRef& c : counters)
                {
                    b.Read(c, Access::CopySrc);
                }
                b.Write(readback, Access::CopyDst);
            },
            [=](rdg::PassContext& ctx)
            {
                for (u32 j = 0; j < u32(counters.size()); ++j)
                {
                    ctx.cmd.CopyBuffer(ctx.Get(readback), (u64(slot) * gpu::kMaxCascades + j) * gpu::kCounterBytes,
                                       ctx.Get(counters[j]), 0, gpu::kCounterBytes);
                }
            });
    }
    ++mFrames;

    // Light the surface voxels and update the radiance cache, every cascade in one dispatch each (y: the
    // cascade): a cascade built whole now whole, the others a budget each (CascadeData::lightBudget,
    // cacheProbes, filled by BeginFrame).
    bool anyRebuilt = false;
    bool anyBuilt = false;
    for (u32 i = 0; i < count; ++i)
    {
        anyRebuilt = anyRebuilt || rebuilt[i];
        anyBuilt = anyBuilt || mCascades[i].built;
    }
    if (!anyBuilt)
    {
        return;
    }
    auto declareTraceReads = [&](rdg::PassBuilder& b, bool voxels)
    {
        for (u32 j = 0; j < count; ++j)
        {
            b.Read(mImported.sdf[j], Access::SRVCompute);
            b.Read(mImported.bricks[j], Access::SRVCompute);
            b.Read(mImported.voxelIndex[j], Access::SRVCompute);
            if (voxels)
            {
                b.Read(mImported.voxels[j], Access::SRVCompute);
            }
        }
    };
    gpu::GiConstants constants{};
    constants.frameData = frameData;
    constants.frameOffset = frameOffset;
    constants.lights = mDevice.GetBindlessSRV(scene.lights);
    constants.input1 = mSettings.radiosityRays;
    constants.input2 = mSettings.radiosityDistance;
    constants.input3 = air::gpu::kInvalidIndex; // the budgets (the second pass: the new voxels)
    u32 capacity = 0;
    for (const Cascade& c : mCascades)
    {
        capacity = std::max(capacity, c.capacity);
    }
    const u32 voxelThreads =
        6u * (anyRebuilt || mRelight ? capacity : std::min(std::max(mSettings.lightingBudget, 64u), capacity));
    const rhi::PipelineHandle light = mLight.Get(mDevice);
    graph.AddPass(
        "ProbeGi.VoxelLight", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            declareTraceReads(b, false);
            for (u32 j = 0; j < count; ++j)
            {
                b.Read(mImported.counters[j], Access::SRVCompute);
                b.Write(mImported.voxels[j], Access::UAVCompute); // radiosity rays read every cascade's light too
                b.Read(mImported.cacheRadiance[j], Access::SRVCompute);
                b.Read(mImported.cacheDistance[j], Access::SRVCompute);
                b.Read(mImported.cacheIrradiance[j], Access::SRVCompute);
            }
        },
        [=](rdg::PassContext& ctx)
        {
            ctx.cmd.SetPipeline(light);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch((voxelThreads + 63) / 64, count, 1);
        });
    // The voxels new where cascades moved, every frame while they are young (only the cascades that moved lately).
    std::vector<u32> freshCascades;
    std::vector<u32> freshIndices;
    for (u32 i = 0; i < count; ++i)
    {
        if (mCascades[i].freshUntil != 0 && mFrames <= mCascades[i].freshUntil)
        {
            freshCascades.push_back(i);
            freshIndices.push_back(mDevice.GetBindlessSRV(mCascades[i].fresh));
        }
    }
    if (!freshCascades.empty())
    {
        graph.AddPass(
            "ProbeGi.VoxelLight.Fresh", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                declareTraceReads(b, false);
                for (u32 j = 0; j < count; ++j)
                {
                    b.Read(mImported.counters[j], Access::SRVCompute);
                    b.Read(mImported.fresh[j], Access::SRVCompute);
                    b.Write(mImported.voxels[j], Access::UAVCompute);
                    b.Read(mImported.cacheRadiance[j], Access::SRVCompute);
                    b.Read(mImported.cacheDistance[j], Access::SRVCompute);
                    b.Read(mImported.cacheIrradiance[j], Access::SRVCompute);
                }
            },
            [=](rdg::PassContext& ctx) mutable
            {
                ctx.cmd.SetPipeline(light);
                for (size_t j = 0; j < freshCascades.size(); ++j)
                {
                    constants.cascade = freshCascades[j];
                    constants.input3 = freshIndices[j];
                    ctx.cmd.PushConstants(&constants, sizeof(constants));
                    ctx.cmd.Dispatch((6u * gpu::kFreshWindow + 63) / 64, 1, 1);
                }
            });
    }

    const u32 probes =
        anyRebuilt || mRelight ? gpu::kCacheProbes : std::clamp(mSettings.cacheProbesPerFrame, 1u, gpu::kCacheProbes);
    const u32 texels = gpu::kCacheTexels * gpu::kCacheTexels;
    rdg::BufferRef rays =
        graph.CreateBuffer({.size = u64(count) * probes * texels * 16,
                            .stride = 16,
                            .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource,
                            .debugName = "ProbeGi.Cache.Rays"});
    gpu::GiConstants cache{};
    cache.frameData = frameData;
    cache.frameOffset = frameOffset;
    cache.probeCount = probes;
    cache.blend = mSettings.cacheBlend;
    const rhi::PipelineHandle trace = mCacheTrace.Get(mDevice);
    graph.AddPass(
        "ProbeGi.Cache.Trace", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            declareTraceReads(b, true);
            b.Write(rays, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            cache.rays = ctx.UAV(rays);
            ctx.cmd.SetPipeline(trace);
            ctx.cmd.PushConstants(&cache, sizeof(cache));
            ctx.cmd.Dispatch(probes, count, 1);
        });
    const rhi::PipelineHandle integrate = mCacheIntegrate.Get(mDevice);
    graph.AddPass(
        "ProbeGi.Cache.Integrate", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(rays, Access::SRVCompute);
            for (u32 j = 0; j < count; ++j)
            {
                b.Write(mImported.cacheRadiance[j], Access::UAVCompute);
                b.Write(mImported.cacheDistance[j], Access::UAVCompute);
                b.Write(mImported.cacheIrradiance[j], Access::UAVCompute);
            }
        },
        [=](rdg::PassContext& ctx) mutable
        {
            cache.rays = ctx.SRV(rays);
            ctx.cmd.SetPipeline(integrate);
            ctx.cmd.PushConstants(&cache, sizeof(cache));
            ctx.cmd.Dispatch(probes, count, 1);
        });
}
} // namespace air::pgi
