// The real-time volumes (docs/architecture.md §8.13) on the CPU, with the GPU's shader code
// (pt::RenderVolumeMarchReference, pt::VolumeMarchProbe): the light a group of emitters sends through a
// point against its integral over them, the moment shadow maps against the transmittance marched
// through the voxels, and the whole method against the path tracer that tracks the volumes, in the
// white furnace and lit by the sun, a sky and embers.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/features/pathtracer/ReferenceRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <utility>
#include <vector>

using namespace air;
using renderer::float3;

namespace
{

// The mean luminance of the image's rows [0, rows).
double MeanLuminance(const Image& image, u32 rows)
{
    double sum = 0.0;
    for (u32 y = 0; y < rows; ++y)
    {
        for (u32 x = 0; x < image.width; ++x)
        {
            const float* p = image.At(x, y);
            sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        }
    }
    return sum / (double(image.width) * rows);
}

bool Near(double value, double expected, double relative, const char* what)
{
    if (std::abs(value - expected) <= relative * std::abs(expected))
    {
        return true;
    }
    std::fprintf(stderr, "  %s = %.4f, expected %.4f +- %.1f%%\n", what, value, expected, relative * 100.0);
    return false;
}

double Phase(double g, double u)
{
    const double d = 1.0 + g * g - 2.0 * g * u;
    return (1.0 - g * g) / (4.0 * std::numbers::pi * d * std::sqrt(d));
}

// The integral over light group `group`'s triangles (one-sided emitters) of their radiance times the
// phase function towards the camera: each triangle cut into n^2 pieces, every piece by its solid angle.
double IntegrateGroup(const renderer::SceneDesc& scene, u32 group, float3 x, float3 d, float g, u32 n)
{
    double sum = 0.0;
    for (const gpu::LightTriangle& t : renderer::BuildLightTriangles(scene))
    {
        if (t.group != group)
        {
            continue;
        }
        const float3 normal = hlsl::normalize(hlsl::cross(t.e1, t.e2));
        const double pieceArea = double(t.area) / (double(n) * n);
        for (u32 i = 0; i < n; ++i)
        {
            for (u32 j = 0; j + i < n; ++j)
            {
                // The upward piece of cell (i, j) of the triangle's barycentric grid, and the downward one.
                for (int flip = 0; flip < 2; ++flip)
                {
                    if (flip == 1 && i + j + 1 >= n)
                    {
                        continue;
                    }
                    const float offset = flip == 0 ? 1.0f / 3.0f : 2.0f / 3.0f;
                    const float3 toPiece =
                        t.p0 + t.e1 * ((float(i) + offset) / float(n)) + t.e2 * ((float(j) + offset) / float(n)) - x;
                    const double distance = hlsl::length(toPiece);
                    const float3 w = toPiece / float(distance);
                    const double cosEmitter = -hlsl::dot(normal, w);
                    if (cosEmitter > 0.0)
                    {
                        sum +=
                            t.radiance.x * pieceArea * cosEmitter / (distance * distance) * Phase(g, hlsl::dot(d, w));
                    }
                }
            }
        }
    }
    return sum;
}

// The scene with one more emissive mesh at `position`, and its light group.
u32 AddEmitter(renderer::SceneDesc& scene, renderer::Mesh mesh, float3 position)
{
    scene.materials.push_back({.name = "Emitter", .baseColor = {0.0f, 0.0f, 0.0f}, .emissionNits = 1000.0f});
    scene.meshes.push_back(std::move(mesh));
    renderer::Instance emitter;
    emitter.name = "Emitter";
    emitter.mesh = u32(scene.meshes.size() - 1);
    emitter.material = u32(scene.materials.size() - 1);
    emitter.transform[0][3] = position.x;
    emitter.transform[1][3] = position.y;
    emitter.transform[2][3] = position.z;
    scene.instances.push_back(emitter);
    const std::vector<gpu::LightTriangle> lights = renderer::BuildLightTriangles(scene);
    return lights[renderer::FirstLightIndices(scene).back()].group;
}

// The smoke over its embers at night, nothing else in the scene.
renderer::SceneDesc EmbersAndSmoke()
{
    renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
    scene.sky.physical = false; // a black environment
    std::erase_if(scene.instances, [](const renderer::Instance& in) { return in.name != "Embers"; });
    return scene;
}

pt::ReferenceSettings Settings(TaskSystem& tasks, u32 size, u32 spp)
{
    pt::ReferenceSettings settings;
    settings.width = size;
    settings.height = size;
    settings.samplesPerPixel = spp;
    settings.maxBounces = 4;
    settings.tasks = &tasks;
    return settings;
}

} // namespace

// A group of a few triangles lights the march exactly, triangle by triangle (each by its solid angle,
// Van Oosterom and Strackee; the phase function at its centre): an eighth of the sphere around the
// point is an eighth of its light. Larger groups are a sphere (closed) or a disc (open) of their
// area, close to the integral over their triangles from a few radii away; the phase function at the
// group's centre is the approximation that shows when it scatters forwards.
TEST(VolumeMarch_GroupLightMatchesItsEmitters)
{
    renderer::SceneDesc scene = airtest::BuiltinScene("cornell_box");
    const float3 x(0.0f, 1.0f, 0.0f);
    // An octant (wound to face x): a solid angle of pi / 2.
    renderer::Mesh octant;
    octant.name = "Octant";
    for (const float3 p : {float3(1.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f), float3(0.0f, 1.0f, 0.0f)})
    {
        octant.vertices.push_back({.position = p, .normal = hlsl::normalize(float3(-1.0f))});
    }
    octant.indices = {0, 1, 2};
    const u32 octantGroup = AddEmitter(scene, std::move(octant), x);
    const u32 octantInstance = u32(scene.instances.size() - 1);
    // A sphere 1.5 m away, and a square of 4 x 4 quads 2 m above.
    const u32 sphereGroup = AddEmitter(scene, renderer::MakeSphere("Ball", 0.3f, 24), x + float3(1.5f, 0.0f, 0.0f));
    renderer::Mesh square;
    square.name = "Square";
    for (u32 i = 0; i <= 4; ++i)
    {
        for (u32 j = 0; j <= 4; ++j)
        {
            square.vertices.push_back({.position = float3(-0.4f + 0.2f * float(i), 0.0f, -0.4f + 0.2f * float(j)),
                                       .normal = {0.0f, -1.0f, 0.0f}});
        }
    }
    for (u32 i = 0; i < 4; ++i)
    {
        for (u32 j = 0; j < 4; ++j)
        {
            const u32 v = i * 5 + j;
            square.indices.insert(square.indices.end(), {v, v + 5, v + 1, v + 1, v + 5, v + 6}); // facing down
        }
    }
    const u32 squareGroup = AddEmitter(scene, std::move(square), x + float3(0.3f, 2.0f, 0.0f));

    const pt::VolumeMarchProbe probe(scene, {});
    const float3 d = hlsl::normalize(float3(0.3f, -0.2f, -1.0f));
    const float radiance =
        renderer::BuildLightTriangles(scene)[renderer::FirstLightIndices(scene)[octantInstance]].radiance.x;
    EXPECT(Near(probe.GroupInScattering(octantGroup, x, d, 0.0f).x, radiance / 8.0, 1e-4, "an octant's light"));
    for (const float g : {0.0f, 0.5f})
    {
        const double tolerance = g == 0.0f ? 0.01 : 0.08;
        for (const float3 view : {d, float3(1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f)})
        {
            EXPECT(Near(probe.GroupInScattering(sphereGroup, x, view, g).x,
                        IntegrateGroup(scene, sphereGroup, x, view, g, 8), tolerance, "a sphere's light"));
            EXPECT(Near(probe.GroupInScattering(squareGroup, x, view, g).x,
                        IntegrateGroup(scene, squareGroup, x, view, g, 16), tolerance, "a square's light"));
        }
    }
}

// The sun's moment shadow map lets through to every point what the volumes' voxels let through,
// within the reconstruction's error (MomentTests.cpp): its texels' rays are marched through a coarser
// level of the bricks, their moments filtered and rounded to half floats, the transmittance
// reconstructed from four of them. Measured: 0.049 on average where the plume shades, at worst 0.26.
TEST(VolumeMarch_MomentShadowsMatchTheVoxels)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
    scene.sky.sunElevation = 25.0f;
    scene.sky.sunAzimuth = -90.0f;
    const pt::VolumeMarchProbe probe(scene, {}, &tasks);
    EXPECT(probe.HasSunMap());
    double error = 0.0;
    double worst = 0.0;
    u32 count = 0;
    for (u32 i = 0; i < 12; ++i)
    {
        for (u32 j = 0; j < 24; ++j)
        {
            for (u32 k = 0; k < 12; ++k)
            {
                const float3 p(-0.55f + 0.1f * float(i), 0.55f + 0.1f * float(j), -0.55f + 0.1f * float(k));
                const double exact = probe.SunTransmittance(p);
                if (exact < 0.99)
                {
                    const double e = std::abs(probe.SunShadow(p) - exact);
                    error += e;
                    worst = std::max(worst, e);
                    ++count;
                }
            }
        }
    }
    EXPECT(count > 500);
    std::fprintf(stderr, "  %u points in the shade: mean error %.4f, worst %.4f\n", count, error / count, worst);
    EXPECT(error / count < 0.06);
    EXPECT(worst < 0.35);
}

// A bed of embers right under a column of smoke has four maps, from the centres of its quadrants (one
// with pathtracer.volumeLightOrigins=1): its centre sees the smoke above through the column's dense
// foot, which the edges of the embers light past. Their mean follows the transmittance towards those
// points, and lets several times more light through to the upper smoke than the centre's map alone.
TEST(VolumeMarch_LargeEmittersLookFromSeveralPoints)
{
    TaskSystem tasks(4);
    const renderer::SceneDesc scene = EmbersAndSmoke();
    const pt::VolumeMarchProbe probe(scene, {}, &tasks);
    pt::VolumeMarchSettings centre;
    centre.lightOrigins = 1;
    const pt::VolumeMarchProbe centreProbe(scene, centre, &tasks);
    EXPECT(probe.GetLightCount() == 1 && probe.GetGroupMapCount(0) == 4 && probe.GetMapCount() == 4);
    EXPECT(centreProbe.GetGroupMapCount(0) == 1);
    double error = 0.0;
    double spread = 0.0;
    double point = 0.0;
    u32 count = 0;
    u32 upper = 0;
    for (u32 i = 0; i < 12; ++i)
    {
        for (u32 j = 0; j < 24; ++j)
        {
            for (u32 k = 0; k < 12; ++k)
            {
                const float3 p(-0.55f + 0.1f * float(i), 0.65f + 0.1f * float(j), -0.55f + 0.1f * float(k));
                const double exact = probe.GroupTransmittance(0, p);
                if (exact > 0.999)
                {
                    continue;
                }
                error += std::abs(probe.GroupVisibility(0, p) - exact);
                ++count;
                if (p.y > 1.6f)
                {
                    spread += probe.GroupVisibility(0, p);
                    point += centreProbe.GroupVisibility(0, p);
                    ++upper;
                }
            }
        }
    }
    EXPECT(count > 500 && upper > 100);
    std::fprintf(stderr, "  %u points in the shade: mean error %.4f; above 1.6 m %.4f, from the centre %.4f\n", count,
                 error / count, spread / upper, point / upper);
    EXPECT(error / count < 0.05);
    EXPECT(spread > 3.0 * point);
}

// In the white furnace (a uniform sky, albedo 1, no lights) the volume vanishes: what the march takes
// out of the light from behind it, its probes' light scattered at every step gives back.
TEST(VolumeMarch_VolumeVanishesInAWhiteFurnace)
{
    TaskSystem tasks(4);
    renderer::SceneDesc scene = airtest::BuiltinScene("smoke");
    scene.instances.resize(1);
    scene.meshes[scene.instances[0].mesh] =
        renderer::MakeQuad("Speck", {-0.01f, -1000.0f, 0.01f}, {0.01f, -1000.0f, 0.01f}, {0.01f, -1000.0f, -0.01f},
                           {-0.01f, -1000.0f, -0.01f}, {0.0f, 1.0f, 0.0f}); // as in IntegratorTests.cpp
    scene.volumes[0].extinction = 8.0f;
    scene.camera.position = {0.0f, 1.9f, 3.0f};
    scene.camera.yaw = 0.0f;
    scene.camera.pitch = 0.0f;
    scene.camera.verticalFov = 1.0f;
    pt::ReferenceSettings settings = Settings(tasks, 24, 1);
    settings.furnace = true;
    settings.volumeEvents = 4096;
    pt::VolumeMarchReferenceSettings march;
    march.frames = 4;
    march.probeFrames = 4;
    const Image image = pt::RenderVolumeMarchReference(scene, scene.camera, settings, march);
    double lowest = 1e30;
    double highest = 0.0;
    for (u32 y = 0; y < image.height; ++y)
    {
        for (u32 x = 0; x < image.width; ++x)
        {
            const float* p = image.At(x, y);
            const double l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            lowest = std::min(lowest, l);
            highest = std::max(highest, l);
        }
    }
    EXPECT(Near(lowest, 0.18, 1e-3, "darkest pixel"));
    EXPECT(Near(highest, 0.18, 1e-3, "brightest pixel"));
}

// Lit, the march gives what the path tracer gives when it tracks the volume: the plume in the sun from
// its side (its own shadow, the sky, the light it scatters within itself from the probes), and at
// night over its embers, nothing else in the scene (their light through four maps; with walls, the
// light the smoke scatters onto them would be missing, which the march leaves out). Measured: -2.9%
// and, above the embers, +0.9%; with the embers' map from their centre alone the night is -44%, with
// probes on a lattice of 4^3 instead of 8^3 the sun's view about -11%.
TEST(VolumeMarch_MatchesTrackedVolumes)
{
    TaskSystem tasks(4);
    for (int view = 0; view < 2; ++view)
    {
        renderer::SceneDesc scene = view == 0 ? airtest::BuiltinScene("smoke") : EmbersAndSmoke();
        if (view == 0)
        {
            for (renderer::Material& material : scene.materials)
            {
                material.emissionNits = 0.0f;
            }
            scene.sky.sunElevation = 25.0f;
            scene.sky.sunAzimuth = -90.0f; // from the left
        }
        scene.camera.position = {0.0f, 1.6f, 2.2f};
        scene.camera.yaw = 0.0f;
        scene.camera.pitch = 0.03f;
        scene.camera.verticalFov = 0.9f;
        pt::ReferenceSettings settings = Settings(tasks, 24, 256);
        const u32 rows = view == 0 ? 24 : 18; // above the embers
        const double tracked = MeanLuminance(pt::RenderReference(scene, scene.camera, settings), rows);
        settings.samplesPerPixel = 8;
        pt::VolumeMarchReferenceSettings march;
        march.frames = 24;
        march.probeFrames = 16;
        const double marched =
            MeanLuminance(pt::RenderVolumeMarchReference(scene, scene.camera, settings, march), rows);
        std::fprintf(stderr, "  view %d: tracked %.5f, marched %.5f (%+.2f%%)\n", view, tracked, marched,
                     (marched / tracked - 1.0) * 100.0);
        EXPECT(tracked > 0.0);
        EXPECT(Near(marched, tracked, 0.05, view == 0 ? "in the sun" : "over the embers"));
    }
}
