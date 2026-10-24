#include "VolumeMarcher.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/BlueNoise.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace air::pt
{

using hlsl::float3;
using rhi::Access;

namespace
{

constexpr Access kRest = Access::SRVCompute;
// Always an array: a texture of one slice would be seen as a plain 2D texture.
constexpr u32 kMapSlices = 1 + gpu::kMaxVolumeLights;
static_assert(kMaxVolumeLights == gpu::kMaxVolumeLights);

float BoxDistance(float3 p, float3 lo, float3 hi)
{
    const float3 d = hlsl::max(hlsl::max(lo - p, p - hi), float3(0.0f));
    return hlsl::length(d);
}

float BoxFarthest(float3 p, float3 lo, float3 hi)
{
    float farthest = 0.0f;
    for (int c = 0; c < 8; ++c)
    {
        const float3 corner((c & 1) != 0 ? hi.x : lo.x, (c & 2) != 0 ? hi.y : lo.y, (c & 4) != 0 ? hi.z : lo.z);
        farthest = std::max(farthest, hlsl::length(corner - p));
    }
    return farthest;
}

// Two unit vectors that make an orthonormal basis with unit w (Duff et al. 2017).
void Perpendicular(float3 w, float3& u, float3& v)
{
    const float s = w.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + w.z);
    const float b = w.x * w.y * a;
    u = float3(1.0f + s * w.x * w.x * a, s * b, -s * w.x);
    v = float3(b, s + w.y * w.y * a, -w.y);
}

void Destroy(rhi::Device& device, rhi::BufferHandle& buffer)
{
    if (buffer.IsValid())
    {
        device.DestroyBuffer(buffer);
    }
    buffer = {};
}

void Destroy(rhi::Device& device, rhi::TextureHandle& texture)
{
    if (texture.IsValid())
    {
        device.DestroyTexture(texture);
    }
    texture = {};
}

} // namespace

VolumeMarcher::VolumeMarcher(rhi::Device& device) : mDevice(device) {}

VolumeMarcher::~VolumeMarcher()
{
    ReleaseShadows();
    Destroy(mDevice, mProbes);
    Destroy(mDevice, mBlueNoise);
    for (renderer::LazyComputePipeline* pipeline : {&mShadowPipeline, &mProbePipeline, &mMarchPipeline})
    {
        pipeline->Destroy(mDevice);
    }
}

void VolumeMarcher::Initialize(const std::filesystem::path& shaderDir)
{
    mShadowPipeline.Load(mDevice, shaderDir, "VolumeShadows.cs", "VolumeMarch.Shadows",
                         sizeof(gpu::VolumeShadowConstants));
    mProbePipeline.Load(mDevice, shaderDir, "VolumeProbes.cs", "VolumeMarch.Probes", sizeof(gpu::VolumeProbeConstants));
    mMarchPipeline.Load(mDevice, shaderDir, "VolumeMarch.cs", "VolumeMarch.March", sizeof(gpu::VolumeMarchConstants));
}

bool VolumeMarcher::IsAvailable() const
{
    return mShadowPipeline.IsAvailable() && mProbePipeline.IsAvailable() && mMarchPipeline.IsAvailable();
}

void VolumeMarcher::ReleaseShadows()
{
    Destroy(mDevice, mB0);
    Destroy(mDevice, mMoments);
    Destroy(mDevice, mOccluders);
    Destroy(mDevice, mMaps);
    Destroy(mDevice, mGroupMaps);
    Destroy(mDevice, mMarchLights);
    mMapsAccess = Access::None;
    mShadowSize = 0;
}

void VolumeMarcher::BeginFrame(rdg::RenderGraph& graph, const renderer::GpuScene& scene,
                               const VolumeMarchSettings& settings, gpu::VolumeShadowsData& shadows)
{
    if (!mBlueNoise.IsValid())
    {
        const std::vector<u16>& noise = renderer::BlueNoise();
        const std::vector<u32> ranks(noise.begin(), noise.end());
        mBlueNoise = renderer::CreateBufferWithData(
            mDevice,
            {.stride = sizeof(u32), .usage = rhi::BufferUsage::ShaderResource, .debugName = "VolumeMarch.BlueNoise"},
            ranks.data(), ranks.size() * sizeof(u32));
    }
    mSettings = settings;
    mSettings.shadowSize = std::clamp(settings.shadowSize, 16u, 2048u);
    mSettings.lights = std::min(settings.lights, gpu::kMaxVolumeLights);
    mSettings.level = std::min(settings.level, air::gpu::kVolumeLevels - 1);
    mSettings.shadowLevel = std::min(settings.shadowLevel, air::gpu::kVolumeLevels - 1);
    mSettings.step = std::max(settings.step, 0.05f);
    mSettings.shadowStep = std::max(settings.shadowStep, 0.05f);
    ++mFrame;

    // The maps show what the scene's version and the shadow settings say; anything else renders them again.
    VolumeMarchSettings shadowKey = mSettings;
    shadowKey.mode = 0;
    shadowKey.step = 0.0f;
    shadowKey.level = 0;
    shadowKey.air = false;
    const bool resize = mShadowSize != mSettings.shadowSize;
    mRenderShadows = resize || scene.version != mShadowVersion || !(shadowKey == mShadowSettings) ||
                     mSettings.everyFrame || !mMaps.IsValid();
    if (resize)
    {
        ReleaseShadows();
        auto make = [&](rhi::Format format, const char* name)
        {
            return mDevice.CreateTexture(
                {.format = format,
                 .width = mSettings.shadowSize,
                 .height = mSettings.shadowSize,
                 .depthOrArraySize = kMapSlices,
                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                 .debugName = name});
        };
        mB0 = make(rhi::Format::R16_FLOAT, "VolumeMarch.ShadowDepth");
        mMoments = make(rhi::Format::RGBA16_FLOAT, "VolumeMarch.ShadowMoments");
        mOccluders = make(rhi::Format::R32_FLOAT, "VolumeMarch.ShadowOccluders");
        mShadowSize = mSettings.shadowSize;
    }
    if (mRenderShadows)
    {
        // The light groups that matter most to the volumes: power over the squared distance to their box.
        float3 lo(1e30f);
        float3 hi(-1e30f);
        for (const air::gpu::VolumeMarchData& v : scene.volumeMarch)
        {
            lo = hlsl::min(lo, v.boundsMin);
            hi = hlsl::max(hi, v.boundsMax);
        }
        std::vector<std::pair<float, u32>> ranked;
        for (u32 i = 0; i < scene.lightGroups.size() && !scene.volumeMarch.empty(); ++i)
        {
            const air::gpu::LightGroup& g = scene.lightGroups[i];
            if (g.power > 0.0f)
            {
                const float d = BoxDistance(g.centre, lo, hi);
                ranked.emplace_back(g.power / (d * d + g.radius * g.radius + 1e-4f), i);
            }
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        mLights.clear();
        for (size_t i = 0; i < ranked.size() && i < mSettings.lights; ++i)
        {
            mLights.push_back(ranked[i].second);
        }

        // A map for the sun over the volumes' box (orthographic), one around each group's centre
        // (octahedral), their depths spanning the box as seen from the light.
        mMapData.clear();
        mSunMap = air::gpu::kInvalidIndex;
        const float margin = 0.02f * std::max(hlsl::length(hi - lo), 1e-3f);
        const bool sun = hlsl::dot(scene.sky.sunRadiance, float3(1.0f)) > 0.0f && !scene.volumeMarch.empty();
        if (sun)
        {
            air::gpu::MomentShadowMap m{};
            m.kind = air::gpu::kMomentMapSun;
            m.axisW = -scene.sky.sunDirection;
            float3 u;
            float3 v;
            Perpendicular(m.axisW, u, v);
            float3 low(1e30f);
            float3 high(-1e30f);
            for (int c = 0; c < 8; ++c)
            {
                const float3 corner((c & 1) != 0 ? hi.x : lo.x, (c & 2) != 0 ? hi.y : lo.y, (c & 4) != 0 ? hi.z : lo.z);
                const float3 projected(hlsl::dot(corner, u), hlsl::dot(corner, v), hlsl::dot(corner, m.axisW));
                low = hlsl::min(low, projected);
                high = hlsl::max(high, projected);
            }
            low = low - float3(margin);
            high = high + float3(margin);
            m.origin = u * low.x + v * low.y + m.axisW * low.z;
            m.axisU = u * (high.x - low.x);
            m.axisV = v * (high.y - low.y);
            m.depthNear = 0.0f;
            m.depthFar = high.z - low.z;
            m.slice = 0;
            m.size = mSettings.shadowSize;
            m.skipInstance = air::gpu::kInvalidIndex;
            mSunMap = 0;
            mMapData.push_back(m);
        }
        std::vector<u32> groupMaps(std::max<size_t>(scene.lightGroups.size(), 1), air::gpu::kInvalidIndex);
        for (u32 group : mLights)
        {
            const air::gpu::LightGroup& g = scene.lightGroups[group];
            air::gpu::MomentShadowMap m{};
            m.kind = air::gpu::kMomentMapPoint;
            m.origin = g.centre;
            m.depthNear = std::max(BoxDistance(g.centre, lo, hi) - margin, 0.0f);
            m.depthFar = BoxFarthest(g.centre, lo, hi) + margin;
            m.slice = u32(mMapData.size());
            m.size = mSettings.shadowSize;
            m.skipInstance = g.instance;
            groupMaps[group] = u32(mMapData.size());
            mMapData.push_back(m);
        }
        Destroy(mDevice, mMaps);
        Destroy(mDevice, mGroupMaps);
        Destroy(mDevice, mMarchLights);
        const air::gpu::MomentShadowMap noMap{};
        mMaps = renderer::CreateBufferWithData(mDevice,
                                               {.stride = sizeof(air::gpu::MomentShadowMap),
                                                .usage = rhi::BufferUsage::ShaderResource,
                                                .debugName = "VolumeMarch.Maps"},
                                               mMapData.empty() ? &noMap : mMapData.data(),
                                               std::max<size_t>(mMapData.size(), 1) * sizeof(noMap));
        mGroupMaps = renderer::CreateBufferWithData(
            mDevice,
            {.stride = sizeof(u32), .usage = rhi::BufferUsage::ShaderResource, .debugName = "VolumeMarch.GroupMaps"},
            groupMaps.data(), groupMaps.size() * sizeof(u32));
        const u32 none = 0;
        mMarchLights = renderer::CreateBufferWithData(
            mDevice,
            {.stride = sizeof(u32), .usage = rhi::BufferUsage::ShaderResource, .debugName = "VolumeMarch.Lights"},
            mLights.empty() ? &none : mLights.data(), std::max<size_t>(mLights.size(), 1) * sizeof(u32));
        mShadowVersion = scene.version;
        mShadowSettings = shadowKey;
    }

    const Access start = mMapsAccess;
    mB0Ref = graph.ImportTexture(mB0, start, kRest);
    mMomentsRef = graph.ImportTexture(mMoments, start, kRest);
    mOccludersRef = graph.ImportTexture(mOccluders, start, kRest);
    mMapsAccess = kRest;
    if (mRenderShadows && !mMapData.empty())
    {
        ++mShadowRenders;
        const rhi::PipelineHandle pipeline = mShadowPipeline.Get(mDevice);
        AIR_CHECK(pipeline.IsValid(), "VolumeMarch.Shadows did not compile");
        gpu::VolumeShadowConstants c{};
        c.maps = mDevice.GetBindlessSRV(mMaps);
        c.volumes = mDevice.GetBindlessSRV(scene.volumeMarchTable);
        c.levels = mDevice.GetBindlessSRV(scene.volumeLevelTable);
        c.volumeCount = u32(scene.volumeMarch.size());
        c.size = mShadowSize;
        c.level = mSettings.shadowLevel;
        c.stepScale = mSettings.shadowStep;
        c.tlas = mDevice.GetBindlessSRV(scene.tlas);
        const u32 count = u32(mMapData.size());
        const rdg::TextureRef b0 = mB0Ref;
        const rdg::TextureRef moments = mMomentsRef;
        const rdg::TextureRef occluders = mOccludersRef;
        graph.AddPass(
            "VolumeMarch.Shadows", rdg::PassFlags::Compute | rdg::PassFlags::AsyncCompute,
            [&](rdg::PassBuilder& b)
            {
                b.Write(b0, Access::UAVCompute);
                b.Write(moments, Access::UAVCompute);
                b.Write(occluders, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                c.b0 = ctx.UAV(b0);
                c.moments = ctx.UAV(moments);
                c.occluders = ctx.UAV(occluders);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&c, sizeof(c));
                ctx.cmd.Dispatch((c.size + 7) / 8, (c.size + 7) / 8, count);
            });
    }

    shadows = {};
    shadows.maps = mDevice.GetBindlessSRV(mMaps);
    shadows.b0 = mDevice.GetBindlessSRV(mB0);
    shadows.moments = mDevice.GetBindlessSRV(mMoments);
    shadows.sunMap = mSunMap;
    shadows.groupMaps = mDevice.GetBindlessSRV(mGroupMaps);
    shadows.lightGroups = mDevice.GetBindlessSRV(scene.lightGroupTable);
    shadows.mapCount = u32(mMapData.size());
    shadows.occluders = mDevice.GetBindlessSRV(mOccluders);
}

void VolumeMarcher::DeclareShadowReads(rdg::PassBuilder& builder) const
{
    builder.Read(mB0Ref, kRest);
    builder.Read(mMomentsRef, kRest);
}

void VolumeMarcher::AddPasses(rdg::RenderGraph& graph, const renderer::GpuScene& scene,
                              const gpu::TraceConstants& trace, rdg::TextureRef surface, rdg::TextureRef accumulation,
                              u32 width, u32 height, u32 samplesPerPixel, u32 debugView, bool reset, bool accumulating,
                              bool furnace)
{
    const u32 volumes = u32(scene.volumeMarch.size());
    AIR_CHECK(volumes > 0 && volumes <= air::gpu::kMaxVolumes, "VolumeMarch: {} volumes", volumes);

    // The probes: new ones for another scene (or volume count), fresh light for a changed scene.
    if (!mProbes.IsValid() || mProbeVolumes != volumes)
    {
        Destroy(mDevice, mProbes);
        mProbes =
            mDevice.CreateTexture({.dimension = rhi::TextureDimension::Tex3D,
                                   .format = rhi::Format::RGBA32_FLOAT,
                                   .width = volumes * gpu::kVolumeProbeAxis,
                                   .height = gpu::kVolumeProbeAxis,
                                   .depthOrArraySize = gpu::kProbeBlocks * gpu::kVolumeProbeAxis,
                                   .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                   .debugName = "VolumeMarch.Probes"});
        mProbesAccess = Access::None;
        mProbeVolumes = volumes;
        mProbeVersion = ~0ull;
    }
    if (scene.version != mProbeVersion)
    {
        mProbeFrames = 0;
        mProbeVersion = scene.version;
    }
    // A running mean of every frame's estimate while the scene stands still and accumulates; when
    // not, one that forgets: 5% of the light is this frame's.
    const float blend =
        mProbeFrames == 0 ? 1.0f : std::max(1.0f / float(mProbeFrames + 1), accumulating ? 0.0f : 0.05f);
    ++mProbeFrames;
    const rdg::TextureRef probes = graph.ImportTexture(mProbes, mProbesAccess, kRest);
    mProbesAccess = kRest;

    const rdg::TextureRef b0 = mB0Ref;
    const rdg::TextureRef moments = mMomentsRef;
    const rdg::TextureRef occluders = mOccludersRef;
    {
        const rhi::PipelineHandle pipeline = mProbePipeline.Get(mDevice);
        AIR_CHECK(pipeline.IsValid(), "VolumeMarch.Probes did not compile");
        gpu::VolumeProbeConstants c{};
        c.frameData = trace.frameData;
        c.frameSlot = trace.frameSlot;
        c.tlas = trace.tlas;
        c.instances = trace.instances;
        c.materials = trace.materials;
        c.lights = trace.lights;
        c.instanceLights = trace.instanceLights;
        c.lightTree = trace.lightTree;
        c.bsdfLut = trace.bsdfLut;
        c.skyTable = trace.skyTable;
        c.volumes = mDevice.GetBindlessSRV(scene.volumeMarchTable);
        c.frame = mFrame;
        c.blend = blend;
        graph.AddPass(
            "VolumeMarch.Probes", rdg::PassFlags::Compute | rdg::PassFlags::AsyncCompute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(b0, kRest);
                b.Read(moments, kRest);
                b.Write(probes, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                c.probes = ctx.UAV(probes);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&c, sizeof(c));
                ctx.cmd.Dispatch(volumes * gpu::kVolumeProbes, 1, 1);
            });
    }

    const rhi::PipelineHandle pipeline = mMarchPipeline.Get(mDevice);
    AIR_CHECK(pipeline.IsValid(), "VolumeMarch.March did not compile");
    gpu::VolumeMarchConstants c{};
    c.frameData = trace.frameData;
    c.frameSlot = trace.frameSlot;
    c.volumes = mDevice.GetBindlessSRV(scene.volumeMarchTable);
    c.levels = mDevice.GetBindlessSRV(scene.volumeLevelTable);
    c.blueNoise = mDevice.GetBindlessSRV(mBlueNoise);
    c.lights = trace.lights;
    c.marchLights = mDevice.GetBindlessSRV(mMarchLights);
    c.marchLightCount = u32(mLights.size());
    c.level = mSettings.level;
    c.stepScale = mSettings.step;
    c.flags =
        (reset ? gpu::kMarchReset : 0u) |
        (mSettings.air && scene.sky.aerialDistance > 0.0f && scene.aerialInscatter.IsValid() ? gpu::kMarchAir : 0u) |
        (debugView == gpu::kDebugIndirect ? gpu::kMarchNoDirect : 0u) |
        (debugView == gpu::kDebugDirect ? gpu::kMarchNoIndirect : 0u) | (furnace ? gpu::kMarchFurnace : 0u);
    c.view = debugView;
    c.samplesPerPixel = samplesPerPixel;
    c.probeVolumes = volumes;
    graph.AddPass(
        "VolumeMarch.March", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(surface, kRest);
            b.Read(b0, kRest);
            b.Read(moments, kRest);
            b.Read(occluders, kRest);
            b.Read(probes, kRest);
            b.Write(accumulation, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            c.surface = ctx.SRV(surface);
            c.accumulation = ctx.UAV(accumulation);
            c.probes = ctx.SRV(probes);
            ctx.cmd.SetPipeline(pipeline);
            ctx.cmd.PushConstants(&c, sizeof(c));
            ctx.cmd.Dispatch((width + gpu::kMarchTile - 1) / gpu::kMarchTile,
                             (height + gpu::kMarchTile - 1) / gpu::kMarchTile, 1);
        });
}

} // namespace air::pt
