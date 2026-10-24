// CPU host for Integrator.hlsli: the scene's tables and BVH (ReferenceScene.h), the path tracer's
// paths through them, and the real-time volumes' probes, which are such paths too.

#include "air/features/pathtracer/ReferenceRenderer.h"

#include "air/core/Assert.h"

#include "ReferenceScene.h"

#include <algorithm>
#include <cmath>

namespace air::pt
{
namespace reference
{
namespace
{

// The shader code becomes member functions; the hooks read the tables.
struct Kernel : SceneHooks
{
    // The light groups the real-time volumes' march lights them with itself (VolumeProbes.hlsl).
    bool VolumeExplicitGroup(uint group) const
    {
        return group < tables.groupMaps.size() && tables.groupMaps[group] != kInvalidIndex;
    }

// NOLINTBEGIN(readability-identifier-naming)
#define AIR_PT_VOLUME 1 // the reference renders everything
#define AIR_PT_PROBE 1  // and the probes of the real-time volumes (kFlagProbe)
#include "Integrator.hlsli"
#include "VolumeProbes.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

} // namespace

std::vector<ProbeLight> TraceVolumeProbes(const SceneTables& tables, const ReferenceSettings& settings,
                                          const SkyData& sky, std::span<const VolumeMarchData> volumes, u32 frames)
{
    const Kernel::IntegratorParams params = MakeParams<Kernel::IntegratorParams>(tables, settings, sky, kFlagProbe);
    const u32 probes = u32(volumes.size()) * kVolumeProbes;
    std::vector<ProbeLight> light(probes);
    auto trace = [&](u32 probe)
    {
        Kernel kernel{{tables}};
        const float3 position = kernel.ProbePosition(volumes[probe / kVolumeProbes], probe % kVolumeProbes);
        double sum[3][4] = {};
        for (u32 frame = 1; frame <= frames; ++frame)
        {
            for (u32 index = 0; index < kVolumeProbeRays; ++index)
            {
                // As VolumeProbes.hlsl: the frame's rotated lattice, and seeds by probe, frame and ray.
                const float3 direction = kernel.ProbeRayDirection(probe, frame, index);
                uint seed = kernel.Pcg(index + kernel.Pcg(probe + kernel.Pcg(frame + 0x50524F42u)));
                const Kernel::PathSample s = kernel.TracePath(position, direction, params, seed);
                const float3 radiance =
                    std::isfinite(s.radiance.x) && std::isfinite(s.radiance.y) && std::isfinite(s.radiance.z)
                        ? s.radiance
                        : float3(0.0f);
                const float4 basis = kernel.ProbeBasis(direction) * (4.0f * Kernel::kPi / float(kVolumeProbeRays));
                const float channels[3] = {radiance.x, radiance.y, radiance.z};
                for (int c = 0; c < 3; ++c)
                {
                    sum[c][0] += double(basis.x * channels[c]);
                    sum[c][1] += double(basis.y * channels[c]);
                    sum[c][2] += double(basis.z * channels[c]);
                    sum[c][3] += double(basis.w * channels[c]);
                }
            }
        }
        auto mean = [&](int c)
        {
            const double n = double(std::max(frames, 1u));
            return float4(float(sum[c][0] / n), float(sum[c][1] / n), float(sum[c][2] / n), float(sum[c][3] / n));
        };
        light[probe] = {mean(0), mean(1), mean(2)};
    };
    if (settings.tasks)
    {
        settings.tasks->ParallelFor(probes, trace);
    }
    else
    {
        for (u32 probe = 0; probe < probes; ++probe)
        {
            trace(probe);
        }
    }
    return light;
}

} // namespace reference

using namespace reference;

Image RenderReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                      const ReferenceSettings& settings)
{
    const std::string problem = renderer::Validate(scene);
    AIR_CHECK(problem.empty(), "reference renderer: {}", problem);

    const SceneTables tables = BuildSceneTables(scene);
    const ReferenceLighting lighting = LightingFor(tables, camera, settings);
    const Kernel::IntegratorParams params = MakeParams<Kernel::IntegratorParams>(tables, settings, lighting.sky, 0u);
    const float exposure = lighting.exposure;

    const float2 size(float(settings.width), float(settings.height));
    const CameraData cameraData = ToCameraData(camera, size.x / size.y);
    Image image(settings.width, settings.height);
    auto renderRow = [&](u32 y)
    {
        Kernel kernel{{tables}};
        for (u32 x = 0; x < settings.width; ++x)
        {
            // Same seeding as PathTracer.hlsl for frame 0.
            uint seed = kernel.Pcg(x + kernel.Pcg(y + kernel.Pcg(0u)));
            double sum[3] = {};
            for (u32 s = 0; s < settings.samplesPerPixel; ++s)
            {
                const float jx = kernel.Rand(seed);
                const float jy = kernel.Rand(seed);
                const float3 direction =
                    kernel.CameraRayDirection(cameraData, float2(float(x) + jx, float(y) + jy), size);
                Kernel::PathSample sample = kernel.TracePath(camera.position, direction, params, seed);
                kernel.AddAerialPerspective(params, direction, sample);
                sum[0] += sample.radiance.x;
                sum[1] += sample.radiance.y;
                sum[2] += sample.radiance.z;
            }
            float* out = image.At(x, y);
            for (int c = 0; c < 3; ++c)
            {
                out[c] = float(sum[c] / settings.samplesPerPixel) * exposure;
            }
            out[3] = 1.0f;
        }
    };
    if (settings.tasks)
    {
        settings.tasks->ParallelFor(settings.height, renderRow);
    }
    else
    {
        for (u32 y = 0; y < settings.height; ++y)
        {
            renderRow(y);
        }
    }
    return image;
}

struct LightTreeProbe::Tables
{
    SceneTables scene;
};

LightTreeProbe::LightTreeProbe(const renderer::SceneDesc& scene) : mTables(std::make_unique<Tables>())
{
    mTables->scene.lights = renderer::BuildLightTriangles(scene);
    mTables->scene.lightTree = renderer::BuildLightTree(mTables->scene.lights);
}

LightTreeProbe::~LightTreeProbe() = default;

u32 LightTreeProbe::GetLightCount() const
{
    return u32(mTables->scene.lights.size());
}

u32 LightTreeProbe::GetNodeCount() const
{
    return u32(mTables->scene.lightTree.size());
}

u32 LightTreeProbe::GetDepth(u32 light) const
{
    return mTables->scene.lights[light].treeDepth;
}

float LightTreeProbe::Pmf(u32 light, renderer::float3 p, renderer::float3 n) const
{
    Kernel kernel{{mTables->scene}}; // the shader functions are not const members
    return kernel.LightTreePmf(light, p, n);
}

u32 LightTreeProbe::Sample(renderer::float3 p, renderer::float3 n, float u, float* pmf) const
{
    Kernel kernel{{mTables->scene}}; // the shader functions are not const members
    const Kernel::LightChoice choice = kernel.SampleLightTree(p, n, u);
    *pmf = choice.pmf;
    return choice.light;
}

} // namespace air::pt
