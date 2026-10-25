// The real-time volumes on the CPU (docs/architecture.md §8.13), with the GPU's shader code: the
// moment shadow maps (VolumeShadows.hlsl), the probes (VolumeProbes.hlsl, whose paths
// ReferenceRenderer.cpp's kernel traces), the surfaces lit through the maps
// (PathTracerVolumeShadows.hlsl) and the ray march over them (VolumeMarch.hlsl).

#include "air/features/pathtracer/ReferenceRenderer.h"

#include "air/core/Assert.h"
#include "air/renderer/BlueNoise.h"
#include "air/renderer/Transfer.h"

#include "ReferenceScene.h"
#include "VolumeShadowLayout.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>

namespace air::pt
{
namespace reference
{
namespace
{

// What the march reads besides the scene: the volumes' bricks, their shadow maps and their probes.
struct MarchTables
{
    std::vector<VolumeMarchData> volumes; // levels: kVolumeLevels entries per volume; ao: the volume's index
    std::vector<VolumeBrickLevel> levels; // their buffer indices: the level's own index
    std::vector<const renderer::VolumeBricks::Level*> levelData;
    std::vector<const renderer::VolumeBricks*> bricks; // per volume
    VolumeShadowLayout shadows;
    u32 mapSize = 0;
    // The maps as the GPU stores them, a slice of mapSize^2 texels per map.
    std::vector<u16> b0;          // R16_FLOAT
    std::vector<u16> moments;     // RGBA16_FLOAT
    std::vector<float> occluders; // R32_FLOAT
    std::vector<float> probes;    // the probe field, RGBA32_FLOAT (VolumeMarchShared.h)
    std::vector<u16> blueNoise;   // 64 x 64 ranks
};

// The shader code becomes member functions; the hooks read the tables.
struct MarchKernel : SceneHooks
{
    const MarchTables& march;

    // MomentShadow.hlsli's: a map's slices, filtered as the GPU filters them.
    float MomentMapB0Sample(MomentShadowMap m, float2 uv) const
    {
        const size_t texels = size_t(march.mapSize) * march.mapSize;
        return renderer::SampleTexture3DClampedR16(march.b0.data() + m.slice * texels, march.mapSize, march.mapSize, 1,
                                                   uv.x, uv.y, 0.5f);
    }
    float4 MomentMapBSample(MomentShadowMap m, float2 uv) const
    {
        const size_t texels = size_t(march.mapSize) * march.mapSize;
        return renderer::SampleTexture3DClamped(march.moments.data() + m.slice * texels * 4, march.mapSize,
                                                march.mapSize, 1, uv.x, uv.y, 0.5f);
    }
    float MomentMapOccluderLoad(MomentShadowMap m, uint x, uint y) const
    {
        return march.occluders[(size_t(m.slice) * march.mapSize + y) * march.mapSize + x];
    }

    // VolumeShadowHooks.hlsli's: the integrator's light through the maps.
    MomentShadowMap LoadMomentMap(uint index) const { return march.shadows.maps[index]; }
    uint VolumeGroupMap(uint group) const { return march.shadows.groupMaps[group]; }
    float VolumeSunShadow(float3 p)
    {
        return march.shadows.sunMap == kInvalidIndex ? 1.0f : MomentShadow(LoadMomentMap(march.shadows.sunMap), p);
    }
    float VolumeGroupShadow(uint group, float3 p, float3 lightPoint)
    {
        const uint first = VolumeGroupMap(group);
        if (first == kInvalidIndex)
        {
            return 1.0f;
        }
        MomentShadowMap m = LoadMomentMap(first);
        float nearest = dot(m.origin - lightPoint, m.origin - lightPoint);
        for (uint k = 1; k < m.count; ++k)
        {
            const MomentShadowMap other = LoadMomentMap(first + k);
            const float d = dot(other.origin - lightPoint, other.origin - lightPoint);
            if (d < nearest)
            {
                nearest = d;
                m = other;
            }
        }
        return MomentShadow(m, p);
    }

    // VolumeBricks.hlsli's: the level's tables.
    uint VolumeIndirectionLoad(VolumeBrickLevel l, uint index) const
    {
        return march.levelData[l.indirection]->indirection[index];
    }
    uint VolumeDistanceLoad(VolumeBrickLevel l, uint index) const
    {
        return march.levelData[l.distance]->distance[index];
    }
    uint VolumeBrickWordLoad(VolumeBrickLevel l, uint index) const { return march.levelData[l.bricks]->bricks[index]; }
    uint VolumeRangeLoad(VolumeBrickLevel l, uint brick) const { return march.levelData[l.ranges]->ranges[brick]; }

    // VolumeMarch.hlsli's.
    VolumeMarchData LoadMarchVolume(uint index) const { return march.volumes[index]; }
    VolumeBrickLevel LoadVolumeLevel(uint index) const { return march.levels[index]; }
    VolumeMarchData LoadRayVolume(uint volume, uint /*slot*/) const { return march.volumes[volume]; }
    VolumeBrickLevel LoadRayLevel(uint volume, uint /*slot*/, uint level) const
    {
        return march.levels[march.volumes[volume].levels + level];
    }
    float VolumeAoSample(VolumeMarchData v, float3 uvw) const
    {
        return renderer::SampleTexture3DClampedR8(march.bricks[v.ao]->ao.data(), v.aoX, v.aoY, v.aoZ, uvw.x, uvw.y,
                                                  uvw.z);
    }
    float4 ProbeFieldSample(float3 uvw) const
    {
        return renderer::SampleTexture3DClampedRGBA32(march.probes.data(), u32(march.volumes.size()) * kVolumeProbeAxis,
                                                      kVolumeProbeAxis, kProbeBlocks * kVolumeProbeAxis, uvw.x, uvw.y,
                                                      uvw.z);
    }
    LightGroup LoadLightGroup(uint group) const { return tables.lightGroups[group]; }
    uint LoadMarchLight(uint i) const { return march.shadows.lights[i]; }

// NOLINTBEGIN(readability-identifier-naming)
#include "air/renderer/shader/MomentShadow.hlsli"
#define AIR_PT_VOLUME_SHADOWS 1 // the surfaces, lit through the maps (PathTracerVolumeShadows.hlsl)
#include "Integrator.hlsli"
#include "VolumeMarch.hlsli"
#include "VolumeProbes.hlsli"
#include "VolumeShadowMarch.hlsli"
#include "air/renderer/shader/VolumeBricks.hlsli"
    // NOLINTEND(readability-identifier-naming)

    // VolumeShadows.hlsl's opaque slice: the first hit along o + t d within (tMin, tMax) past the
    // triangles of instance `skip`, tMax without one.
    float FirstOpaqueHit(float3 o, float3 d, float tMin, float tMax, uint skip) const
    {
        HitInfo hit{};
        return tables.bvh->Intersect(o, d, tMax, false, hit, tMin, skip) ? hit.t : tMax;
    }

    float OpaqueDepth(MomentShadowMap m, float2 uv)
    {
        float3 o;
        float3 d;
        MomentMapRay(m, uv, o, d);
        const float reach = m.depthFar + 0.5f * (m.depthFar - m.depthNear);
        if (m.kind == kMomentMapSun && FirstOpaqueHit(o, -d, 0.0f, 1e30f, kInvalidIndex) < 1e30f)
        {
            return kMomentOccluded;
        }
        const float t = FirstOpaqueHit(o, d, m.kind == kMomentMapSun ? 0.0f : 1e-4f, reach, m.skipInstance);
        return t < reach ? MomentDepth(m, t) : kMomentNoOccluder;
    }
};

float Halton(u32 index, u32 base)
{
    float f = 1.0f;
    float r = 0.0f;
    for (u32 i = index; i > 0; i /= base)
    {
        f /= float(base);
        r += f * float(i % base);
    }
    return r;
}

void ParallelFor(TaskSystem* tasks, u32 count, const std::function<void(u32)>& fn)
{
    if (tasks)
    {
        tasks->ParallelFor(count, fn);
        return;
    }
    for (u32 i = 0; i < count; ++i)
    {
        fn(i);
    }
}

// The volumes as the GPU's tables hold them (GpuScene::volumeMarch, volumeLevels) and their lights' maps.
MarchTables BuildMarchTables(const SceneTables& tables, const VolumeMarchSettings& settings)
{
    MarchTables march;
    for (const renderer::VolumeResources& volume : tables.volumes)
    {
        VolumeMarchData data = volume.bricks.data;
        data.levels = u32(march.levels.size());
        data.ao = u32(march.volumes.size());
        march.volumes.push_back(data);
        march.bricks.push_back(&volume.bricks);
        for (const renderer::VolumeBricks::Level& level : volume.bricks.levels)
        {
            VolumeBrickLevel l = level.data;
            const u32 self = u32(march.levels.size());
            l.indirection = self;
            l.bricks = self;
            l.ranges = self;
            l.distance = self;
            march.levels.push_back(l);
            march.levelData.push_back(&level);
        }
    }
    march.shadows = LayoutVolumeShadows(
        march.volumes, tables.lightGroups,
        std::span(tables.lights.data(), tables.lightPower > 0.0 ? tables.lights.size() : 0), tables.sky.data, settings);
    march.mapSize = settings.shadowSize;
    march.blueNoise = renderer::BlueNoise();
    return march;
}

// VolumeShadows.hlsl: every texel of every map, kept as the GPU keeps it.
void RenderMomentMaps(const SceneTables& tables, MarchTables& march, const VolumeMarchSettings& settings,
                      TaskSystem* tasks)
{
    const u32 n = march.mapSize;
    const size_t texels = size_t(n) * n;
    const size_t slices = std::max<size_t>(march.shadows.maps.size(), 1);
    march.b0.assign(slices * texels, 0);
    march.moments.assign(slices * texels * 4, 0);
    march.occluders.assign(slices * texels, kMomentNoOccluder);
    const u32 volumes = u32(march.volumes.size());
    ParallelFor(tasks, u32(march.shadows.maps.size()) * n,
                [&](u32 row)
                {
                    MarchKernel kernel{{tables}, march};
                    const MomentShadowMap m = march.shadows.maps[row / n];
                    const u32 y = row % n;
                    for (u32 x = 0; x < n; ++x)
                    {
                        const float2 uv((float(x) + 0.5f) / float(n), (float(y) + 0.5f) / float(n));
                        float b0;
                        float4 b;
                        kernel.MarchShadowTexel(m, uv, volumes, settings.shadowLevel, settings.shadowStep, b0, b);
                        const size_t t = m.slice * texels + size_t(y) * n + x;
                        march.b0[t] = renderer::FloatToHalf(b0);
                        march.moments[t * 4 + 0] = renderer::FloatToHalf(b.x);
                        march.moments[t * 4 + 1] = renderer::FloatToHalf(b.y);
                        march.moments[t * 4 + 2] = renderer::FloatToHalf(b.z);
                        march.moments[t * 4 + 3] = renderer::FloatToHalf(b.w);
                        march.occluders[t] = kernel.OpaqueDepth(m, uv);
                    }
                });
}

// The probe field (VolumeProbes.hlsl): the light the probes hold, and each volume's occlusion at them.
void FillProbeField(const SceneTables& tables, MarchTables& march, const ReferenceSettings& settings,
                    const SkyData& sky, u32 frames)
{
    const std::vector<ProbeLight> light = TraceVolumeProbes(tables, settings, sky, march.volumes, frames);
    const u32 width = u32(march.volumes.size()) * kVolumeProbeAxis;
    const u32 height = kVolumeProbeAxis;
    march.probes.assign(size_t(width) * height * kProbeBlocks * kVolumeProbeAxis * 4, 0.0f);
    MarchKernel kernel{{tables}, march};
    for (u32 volume = 0; volume < march.volumes.size(); ++volume)
    {
        const VolumeMarchData& v = march.volumes[volume];
        for (u32 local = 0; local < kVolumeProbes; ++local)
        {
            const ProbeLight& p = light[volume * kVolumeProbes + local];
            const float occlusion = kernel.VolumeOcclusion(v, kernel.ProbePosition(v, local));
            const float4 blocks[kProbeBlocks] = {p.red, p.green, p.blue, float4(occlusion, 0.0f, 0.0f, 0.0f)};
            for (u32 block = 0; block < kProbeBlocks; ++block)
            {
                const u32 x = kernel.ProbeTexelX(volume, local);
                const u32 y = kernel.ProbeTexelY(local);
                const u32 z = kernel.ProbeTexelZ(local, block);
                float* texel = &march.probes[((size_t(z) * height + y) * width + x) * 4];
                texel[0] = blocks[block].x;
                texel[1] = blocks[block].y;
                texel[2] = blocks[block].z;
                texel[3] = blocks[block].w;
            }
        }
    }
}

// What the volumes let through along p + t d for t < reach: their finest bricks marched at a quarter
// voxel.
float MarchedTransmittance(const SceneTables& tables, const MarchTables& march, float3 p, float3 d, float reach)
{
    MarchKernel kernel{{tables}, march};
    double depth = 0.0;
    for (const VolumeMarchData& v : march.volumes)
    {
        float t0;
        float t1;
        if (!kernel.BoxSegment(p, d, v.boundsMin, v.boundsMax, 0.0f, reach, t0, t1))
        {
            continue;
        }
        const VolumeBrickLevel l = march.levels[v.levels];
        const float dt = 0.25f * l.voxelSize;
        for (float t = t0 + 0.5f * dt; t < t1; t += dt)
        {
            depth += double(v.extinction * kernel.BrickDensity(l, p + d * t) * dt);
        }
    }
    return float(std::exp(-depth));
}

} // namespace
} // namespace reference

using namespace reference;

Image RenderVolumeMarchReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                                 const ReferenceSettings& settings, const VolumeMarchReferenceSettings& march)
{
    const std::string problem = renderer::Validate(scene);
    AIR_CHECK(problem.empty(), "volume march reference: {}", problem);
    SceneTables tables = BuildSceneTables(scene);
    AIR_CHECK(!tables.volumes.empty(), "volume march reference: the scene has no volumes");
    const VolumeMarchSettings marchSettings = ClampVolumeMarchSettings(march.march);
    const ReferenceLighting lighting = LightingFor(tables, camera, settings);
    MarchTables marchTables = BuildMarchTables(tables, marchSettings);
    tables.groupMaps = marchTables.shadows.groupMaps;
    RenderMomentMaps(tables, marchTables, marchSettings, settings.tasks);
    FillProbeField(tables, marchTables, settings, lighting.sky, march.probeFrames);

    using Kernel = MarchKernel;
    const Kernel::IntegratorParams params = MakeParams<Kernel::IntegratorParams>(tables, settings, lighting.sky, 0u);
    const bool air = marchSettings.air && lighting.sky.aerialDistance > 0.0f && !tables.sky.aerialInscatter.empty();
    Kernel::MarchSetup setup{};
    setup.sky = lighting.sky;
    setup.sunMap = marchTables.shadows.sunMap;
    setup.lightCount = u32(marchTables.shadows.lights.size());
    setup.probeVolumes = u32(marchTables.volumes.size());
    setup.level = marchSettings.level;
    setup.stepScale = marchSettings.step;
    setup.flags = (air ? kMarchAir : 0u) | (settings.furnace ? kMarchFurnace : 0u);

    const u32 width = settings.width;
    const u32 height = settings.height;
    const u32 spp = std::max(settings.samplesPerPixel, 1u);
    const u32 frames = std::max(march.frames, 1u);
    const float2 size(float(settings.width), float(settings.height));
    const CameraData cameraData = ToCameraData(camera, size.x / size.y);
    const u32 tilesX = (width + kMarchTile - 1) / kMarchTile;
    const u32 tilesY = (height + kMarchTile - 1) / kMarchTile;
    constexpr u32 kTilePixels = kMarchTile * kMarchTile;
    constexpr u32 kNoise = renderer::kBlueNoiseSize;
    Image image(width, height);
    ParallelFor(
        settings.tasks, tilesX * tilesY,
        [&](u32 tile)
        {
            Kernel kernel{{tables}, marchTables};
            const u32 x0 = (tile % tilesX) * kMarchTile;
            const u32 y0 = (tile / tilesX) * kMarchTile;
            double sum[kTilePixels][3] = {};
            float3 directions[kTilePixels];
            float3 surfaces[kTilePixels];
            float depths[kTilePixels];
            for (u32 frame = 0; frame < frames; ++frame)
            {
                // The frame's sub-pixel offset (Halton, as the pathtracer feature accumulates) and the
                // trace's samples along each pixel's ray, then the volumes the tile's rays cross in
                // front of their surfaces (VolumeMarch.hlsl).
                const float2 jitter(Halton(frame + 1, 2) - 0.5f, Halton(frame + 1, 3) - 0.5f);
                u32 mask = 0;
                for (u32 i = 0; i < kTilePixels; ++i)
                {
                    const u32 x = x0 + i % kMarchTile;
                    const u32 y = y0 + i / kMarchTile;
                    if (x >= width || y >= height)
                    {
                        continue;
                    }
                    const float3 d =
                        kernel.CameraRayDirection(cameraData, float2(float(x), float(y)) + jitter + 0.5f, size);
                    uint seed = kernel.Pcg(x + kernel.Pcg(y + kernel.Pcg(frame)));
                    float3 total(0.0f);
                    float tSurface = 1e30f;
                    for (u32 s = 0; s < spp; ++s)
                    {
                        Kernel::PathSample sample = kernel.TracePath(camera.position, d, params, seed);
                        kernel.AddAerialPerspective(params, d, sample);
                        total += sample.radiance;
                        if (s == 0 && sample.hit)
                        {
                            tSurface = sample.hitDistance;
                        }
                    }
                    directions[i] = d;
                    surfaces[i] = total;
                    depths[i] = tSurface;
                    for (u32 v = 0; v < marchTables.volumes.size(); ++v)
                    {
                        float t0;
                        float t1;
                        if (kernel.MarchVolumeSegment(marchTables.volumes[v], camera.position, d, tSurface, t0, t1))
                        {
                            mask |= 1u << v;
                        }
                    }
                }
                u32 tileIndices = 0;
                u32 tileCount = 0;
                for (u32 bits = mask; bits != 0 && tileCount < kMarchMaxVolumes; bits &= bits - 1)
                {
                    tileIndices |= u32(std::countr_zero(bits)) << (8 * tileCount);
                    ++tileCount;
                }

                // Every ray through its volumes, over its samples.
                for (u32 i = 0; i < kTilePixels; ++i)
                {
                    const u32 x = x0 + i % kMarchTile;
                    const u32 y = y0 + i / kMarchTile;
                    if (x >= width || y >= height)
                    {
                        continue;
                    }
                    const float3 d = directions[i];
                    Kernel::MarchRayVolumes rv{};
                    for (u32 k = 0; k < tileCount; ++k)
                    {
                        const u32 index = (tileIndices >> (8 * k)) & 0xFFu;
                        float t0;
                        float t1;
                        if (kernel.MarchVolumeSegment(marchTables.volumes[index], camera.position, d, depths[i], t0,
                                                      t1))
                        {
                            rv.indices |= index << (8 * rv.count);
                            rv.slots |= k << (8 * rv.count);
                            rv.tIn = kernel.SetLane4(rv.tIn, rv.count, t0);
                            rv.tOut = kernel.SetLane4(rv.tOut, rv.count, t1);
                            ++rv.count;
                        }
                    }
                    const uint rank = marchTables.blueNoise[(y % kNoise) * kNoise + x % kNoise];
                    Kernel::MarchSetup ray = setup;
                    ray.lightSample = kernel.MarchLightSample(rank, frame);
                    kernel.MarchChooseMaps(ray);
                    const Kernel::MarchResult r =
                        kernel.MarchRay(ray, rv, camera.position, d, depths[i], kernel.MarchJitter(rank, frame));
                    float3 value = r.radiance * float(spp) + surfaces[i] * r.transmittance;
                    if (air && r.transmittance < 1.0f)
                    {
                        // The air in front of the volumes over their light and over what they let through.
                        const Kernel::Aerial a = kernel.AerialPerspective(setup.sky, d, r.depth);
                        value = (a.inscatter * (1.0f - r.transmittance) + a.transmittance * r.radiance) * float(spp) +
                                surfaces[i] * r.transmittance;
                    }
                    sum[i][0] += value.x;
                    sum[i][1] += value.y;
                    sum[i][2] += value.z;
                }
            }
            const double scale = double(lighting.exposure) / (double(frames) * spp);
            for (u32 i = 0; i < kTilePixels; ++i)
            {
                const u32 x = x0 + i % kMarchTile;
                const u32 y = y0 + i / kMarchTile;
                if (x < width && y < height)
                {
                    float* out = image.At(x, y);
                    for (int c = 0; c < 3; ++c)
                    {
                        out[c] = float(sum[i][c] * scale);
                    }
                    out[3] = 1.0f;
                }
            }
        });
    return image;
}

struct VolumeMarchProbe::State
{
    SceneTables tables;
    MarchTables march;
};

VolumeMarchProbe::VolumeMarchProbe(const renderer::SceneDesc& scene, const VolumeMarchSettings& settings,
                                   TaskSystem* tasks)
    : mState(std::make_unique<State>())
{
    const std::string problem = renderer::Validate(scene);
    AIR_CHECK(problem.empty(), "volume march probe: {}", problem);
    const VolumeMarchSettings clamped = ClampVolumeMarchSettings(settings);
    mState->tables = BuildSceneTables(scene);
    mState->march = BuildMarchTables(mState->tables, clamped);
    RenderMomentMaps(mState->tables, mState->march, clamped, tasks);
}

VolumeMarchProbe::~VolumeMarchProbe() = default;

u32 VolumeMarchProbe::GetMapCount() const
{
    return u32(mState->march.shadows.maps.size());
}

u32 VolumeMarchProbe::GetLightCount() const
{
    return u32(mState->march.shadows.lights.size());
}

bool VolumeMarchProbe::HasSunMap() const
{
    return mState->march.shadows.sunMap != kInvalidIndex;
}

float VolumeMarchProbe::SunShadow(renderer::float3 p) const
{
    MarchKernel kernel{{mState->tables}, mState->march}; // the shader functions are not const members
    return kernel.VolumeSunShadow(p);
}

float VolumeMarchProbe::SunVisibility(renderer::float3 p) const
{
    MarchKernel kernel{{mState->tables}, mState->march};
    const u32 map = mState->march.shadows.sunMap;
    return map == kInvalidIndex ? 1.0f : kernel.MomentLightVisibility(kernel.LoadMomentMap(map), p);
}

renderer::float3 VolumeMarchProbe::GroupInScattering(u32 group, renderer::float3 x, renderer::float3 d,
                                                     float anisotropy) const
{
    MarchKernel kernel{{mState->tables}, mState->march};
    return kernel.GroupInScattering(mState->tables.lightGroups[group], x, d, float4(1.0f, 0.0f, 0.0f, 0.0f),
                                    float4(anisotropy, 0.0f, 0.0f, 0.0f), 1u);
}

namespace
{

// A march at x along d through every volume, lit by the sun and the light groups through map `sample`
// of each (without the probes, which a probe does not fill).
struct ProbeStep
{
    MarchKernel kernel;
    MarchKernel::MarchSetup setup{};
    MarchKernel::MarchRayVolumes rv{};

    ProbeStep(const SceneTables& tables, const MarchTables& march, u32 sample) : kernel{{tables}, march}
    {
        setup.sky = tables.sky.data;
        setup.sunMap = march.shadows.sunMap;
        setup.lightCount = u32(march.shadows.lights.size());
        setup.probeVolumes = u32(march.volumes.size());
        setup.stepScale = 1.0f;
        setup.flags = kMarchNoIndirect;
        setup.lightSample = sample;
        kernel.MarchChooseMaps(setup);
        for (u32 v = 0; v < march.volumes.size() && v < kMarchMaxVolumes; ++v)
        {
            rv.indices |= v << (8 * v);
            rv.slots |= v << (8 * v);
            rv.tIn = kernel.SetLane4(rv.tIn, v, 0.0f);
            rv.tOut = kernel.SetLane4(rv.tOut, v, 1e30f);
            ++rv.count;
        }
    }
};

} // namespace

renderer::float3 VolumeMarchProbe::StepLight(renderer::float3 x, renderer::float3 d, u32 sample) const
{
    ProbeStep step(mState->tables, mState->march, sample);
    const MarchKernel::MarchPoint p = step.kernel.MarchPointAt(step.setup, step.rv, x, d, 1.0f, 1e30f, false);
    return step.kernel.MarchDirect(step.setup, step.rv, p, x, d);
}

renderer::float3 VolumeMarchProbe::SharedStepLight(renderer::float3 x, renderer::float3 d, u32 sample) const
{
    ProbeStep step(mState->tables, mState->march, sample);
    bool valid = false;
    const float3 light = step.kernel.MarchLightAt(step.setup, step.rv, x, d, 1.0f, false, valid);
    return valid ? light * Scattering(x) : float3(0.0f);
}

float VolumeMarchProbe::Scattering(renderer::float3 x) const
{
    ProbeStep step(mState->tables, mState->march, 0);
    const MarchKernel::MarchPoint p =
        step.kernel.MarchPointAt(step.setup, step.rv, x, float3(0.0f, 0.0f, 1.0f), 1.0f, 1e30f, false);
    return p.sigmaS.x + p.sigmaS.y + p.sigmaS.z + p.sigmaS.w;
}

float VolumeMarchProbe::SunTransmittance(renderer::float3 p) const
{
    return MarchedTransmittance(mState->tables, mState->march, p, mState->tables.sky.data.sunDirection, 1e30f);
}

float VolumeMarchProbe::GroupVisibility(u32 group, renderer::float3 p) const
{
    MarchKernel kernel{{mState->tables}, mState->march};
    const u32 first = mState->march.shadows.groupMaps[group];
    if (first == kInvalidIndex)
    {
        return 1.0f;
    }
    // The mean over the maps, as the frames read them in turn.
    const u32 count = mState->march.shadows.maps[first].count;
    float sum = 0.0f;
    for (u32 k = 0; k < count; ++k)
    {
        sum += kernel.GroupVisibility(first, p, k);
    }
    return sum / float(count);
}

u32 VolumeMarchProbe::GetGroupMapCount(u32 group) const
{
    const u32 first = mState->march.shadows.groupMaps[group];
    return first == kInvalidIndex ? 0u : mState->march.shadows.maps[first].count;
}

float VolumeMarchProbe::GroupTransmittance(u32 group, renderer::float3 p) const
{
    const u32 first = mState->march.shadows.groupMaps[group];
    if (first == kInvalidIndex)
    {
        return 1.0f;
    }
    const u32 count = mState->march.shadows.maps[first].count;
    double sum = 0.0;
    for (u32 k = 0; k < count; ++k)
    {
        const float3 toLight = mState->march.shadows.maps[first + k].origin - p;
        sum += MarchedTransmittance(mState->tables, mState->march, p, normalize(toLight), length(toLight));
    }
    return float(sum / count);
}

} // namespace air::pt
