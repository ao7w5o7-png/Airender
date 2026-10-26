// The scene's volume (renderer::BuildVolume, shader/VolumeLookup.hlsli): placement and padding, the
// majorants bound the density everywhere a ray looks, delta and ratio tracking estimate the
// transmittance a fine ray march computes, averaging down keeps the mass, and the phase function is
// sampled exactly.

#include "TestFramework.h"

#include "air/renderer/Clouds.h"
#include "air/renderer/Transfer.h"
#include "air/renderer/Volume.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

using namespace air;
using namespace air::hlsl;
using namespace air::gpu;

namespace
{

// VolumeLookup.hlsli over resources in memory, as the CPU reference renderer reads them.
struct VolumeReader
{
    const renderer::VolumeResources& volume;

    float VolumeDensitySample(VolumeData v, float3 uvw) const
    {
        return renderer::SampleTexture3DClampedR16(volume.density.data(), v.sizeX + 2, v.sizeY + 2, v.sizeZ + 2, uvw.x,
                                                   uvw.y, uvw.z);
    }
    float VolumeMajorantLoad(VolumeData /*v*/, uint cell) const { return volume.majorants[cell]; }
    // CloudLookup.hlsli (the DDA axis, Henyey-Greenstein) wants its hooks too.
    float CloudBaseSample(CloudData /*c*/, float3 /*uvw*/) const { return 0.0f; }
    float CloudDetailSample(CloudData /*c*/, float3 /*uvw*/) const { return 0.0f; }
    float CloudMajorantLoad(CloudData /*c*/, uint /*cell*/) const { return 0.0f; }

    // Sampling.hlsli's generator.
    static uint Pcg(uint v)
    {
        const uint state = v * 747796405u + 2891336453u;
        const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    static float Rand(uint& seed)
    {
        seed = Pcg(seed);
        return float(seed >> 8) * (1.0f / 16777216.0f);
    }
    // Three uniforms in order (a constructor's arguments are unsequenced).
    static float3 Rand3(uint& seed)
    {
        const float x = Rand(seed);
        const float y = Rand(seed);
        return float3(x, y, Rand(seed));
    }

// NOLINTBEGIN(readability-identifier-naming)
#include "air/renderer/shader/CloudLookup.hlsli"
#include "air/renderer/shader/VolumeLookup.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

bool Near(double value, double expected, double tolerance, const char* what)
{
    if (std::abs(value - expected) <= tolerance)
    {
        return true;
    }
    std::fprintf(stderr, "  %s = %.6g, expected %.6g +- %.3g\n", what, value, expected, tolerance);
    return false;
}

const renderer::VolumeResources& Plume()
{
    static const renderer::VolumeResources volume = renderer::BuildVolume({.enabled = true, .extinction = 30.0f});
    return volume;
}

float3 RandomDirection(uint& seed)
{
    const float z = VolumeReader::Rand(seed) * 2.0f - 1.0f;
    const float phi = VolumeReader::Rand(seed) * 6.2831853f;
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return float3(r * std::cos(phi), z, r * std::sin(phi));
}

} // namespace

TEST(Volume_PlumeIsPlacedAndPadded)
{
    const renderer::VolumeResources& volume = Plume();
    const VolumeData& v = volume.data;
    EXPECT(volume.error.empty());
    EXPECT(v.enabled == 1u && v.sizeX == 64u && v.sizeY == 128u && v.sizeZ == 64u);
    EXPECT(Near(v.voxelSize, 0.025, 1e-7, "voxel size"));
    // Its bottom is centred on the origin.
    EXPECT(Near(v.origin.x, -0.8, 1e-6, "origin x") && Near(v.origin.y, 0.0, 1e-6, "origin y") &&
           Near(v.origin.z, -0.8, 1e-6, "origin z"));
    const u32 px = v.sizeX + 2;
    const u32 py = v.sizeY + 2;
    const u32 pz = v.sizeZ + 2;
    EXPECT(volume.density.size() == size_t(px) * py * pz);
    EXPECT(volume.majorants.size() == size_t(v.cellsX) * v.cellsY * v.cellsZ);
    EXPECT(v.cellsX == (px + kVolumeCell - 1) / kVolumeCell && v.cellsY == (py + kVolumeCell - 1) / kVolumeCell);
    // The padding is empty; the plume is not.
    double inside = 0.0;
    bool paddingEmpty = true;
    for (u32 z = 0; z < pz; ++z)
    {
        for (u32 y = 0; y < py; ++y)
        {
            for (u32 x = 0; x < px; ++x)
            {
                const float d = renderer::HalfToFloat(volume.density[(size_t(z) * py + y) * px + x]);
                const bool border = x == 0 || y == 0 || z == 0 || x == px - 1 || y == py - 1 || z == pz - 1;
                paddingEmpty = paddingEmpty && (!border || d == 0.0f);
                inside += border ? 0.0 : d;
            }
        }
    }
    EXPECT(paddingEmpty);
    EXPECT(inside > 1000.0);
    // Sized and moved: the longest side fits `size`, the bottom centred on `position`.
    const renderer::VolumeResources placed =
        renderer::BuildVolume({.enabled = true, .position = {1.0f, 2.0f, 3.0f}, .size = 6.4f});
    EXPECT(Near(placed.data.voxelSize, 0.05, 1e-7, "scaled voxel size"));
    EXPECT(Near(placed.data.origin.x, 1.0 - 1.6, 1e-5, "placed x") &&
           Near(placed.data.origin.y, 2.0, 1e-6, "placed y") &&
           Near(placed.data.origin.z, 3.0 - 1.6, 1e-5, "placed z"));
}

TEST(Volume_MajorantsBoundTheDensity)
{
    const renderer::VolumeResources& volume = Plume();
    const VolumeData& v = volume.data;
    VolumeReader reader{volume};
    // Along random rays through the box, the density never exceeds the majorant of the segment the
    // DDA gives, at the segments' ends included.
    uint seed = 5u;
    int checked = 0;
    for (int ray = 0; ray < 3000; ++ray)
    {
        const float3 o = VolumeReader::Rand3(seed) * 4.0f - float3(2.0f, 0.5f, 2.0f);
        const float3 d = RandomDirection(seed);
        VolumeReader::VolumeMarch m;
        if (!reader.BeginVolumeMarch(v, o, d, 1e30f, m))
        {
            continue;
        }
        float t0 = 0.0f;
        float t1 = 0.0f;
        float majorant = 0.0f;
        for (int segment = 0; segment < 256 && reader.NextVolumeSegment(v, m, t0, t1, majorant); ++segment)
        {
            for (int k = 0; k < 8; ++k)
            {
                const float t = t0 + (t1 - t0) * (k == 0 ? 0.0f : (k == 7 ? 1.0f : VolumeReader::Rand(seed)));
                const float sigma = v.extinction * reader.VolumeDensity(v, o + d * t);
                if (sigma > majorant)
                {
                    std::fprintf(stderr, "  density %g above majorant %g at t %g of [%g, %g]\n", sigma, majorant, t, t0,
                                 t1);
                }
                EXPECT(sigma <= majorant);
                ++checked;
            }
        }
    }
    EXPECT(checked > 20000);
}

// Ratio tracking's mean and delta tracking's chance to pass are both exp(-optical depth), which a
// fine ray march gives.
TEST(Volume_TrackingMatchesARayMarch)
{
    const renderer::VolumeResources& volume = Plume();
    const VolumeData& v = volume.data;
    VolumeReader reader{volume};
    uint seed = 17u;
    int tested = 0;
    for (int attempt = 0; attempt < 400 && tested < 8; ++attempt)
    {
        // From around the plume through a point inside it.
        const float3 u = VolumeReader::Rand3(seed);
        const float3 target(u.x * 0.6f - 0.3f, 0.2f + u.y * 2.6f, u.z * 0.6f - 0.3f);
        const float3 d = RandomDirection(seed);
        const float3 o = target - d * 4.0f;
        constexpr double kStep = 0.001;
        double depth = 0.0;
        for (double t = 0.5 * kStep; t < 8.0; t += kStep)
        {
            depth += v.extinction * reader.VolumeDensity(v, o + d * float(t)) * kStep;
        }
        if (depth < 0.1 || depth > 3.0)
        {
            continue; // nearly clear or opaque: a weak test
        }
        ++tested;
        const double expected = std::exp(-depth);
        constexpr int kSamples = 20000;
        double ratio = 0.0;
        int passed = 0;
        for (int i = 0; i < kSamples; ++i)
        {
            ratio += reader.VolumeTransmittance(v, o, d, 1e30f, seed);
            float tHit = 0.0f;
            passed += reader.VolumeCollision(v, o, d, 1e30f, seed, tHit) ? 0 : 1;
        }
        const double sigma = std::sqrt(expected * (1.0 - expected) / kSamples);
        EXPECT(Near(ratio / kSamples, expected, 5.0 * sigma + 2e-3, "ratio tracking"));
        EXPECT(Near(double(passed) / kSamples, expected, 5.0 * sigma + 2e-3, "delta tracking"));
        // Over part of the way: a surface at half the distance stops both.
        const double half = [&]
        {
            double sum = 0.0;
            for (double t = 0.5 * kStep; t < 4.0; t += kStep)
            {
                sum += v.extinction * reader.VolumeDensity(v, o + d * float(t)) * kStep;
            }
            return std::exp(-sum);
        }();
        double partial = 0.0;
        for (int i = 0; i < kSamples; ++i)
        {
            partial += reader.VolumeTransmittance(v, o, d, 4.0f, seed);
        }
        EXPECT(Near(partial / kSamples, half, 5.0 * std::sqrt(half * (1.0 - half) / kSamples) + 2e-3,
                    "ratio tracking to a surface"));
    }
    EXPECT(tested == 8);
}

// Averaging blocks of voxels down keeps the mass (density times volume); voxels beyond the grid's
// edge count as empty.
TEST(Volume_DownsampleKeepsTheMass)
{
    renderer::VolumeGrid grid;
    grid.sizeX = 13;
    grid.sizeY = 7;
    grid.sizeZ = 10;
    grid.voxelSize = 0.1f;
    grid.origin = float3(1.0f, 2.0f, 3.0f);
    uint seed = 3u;
    double mass = 0.0;
    for (u32 i = 0; i < grid.sizeX * grid.sizeY * grid.sizeZ; ++i)
    {
        grid.density.push_back(VolumeReader::Rand(seed));
        mass += grid.density.back();
    }
    mass *= 0.001;
    for (u32 factor : {2u, 3u, 4u})
    {
        const renderer::VolumeGrid coarse = renderer::Downsample(grid, factor);
        EXPECT(coarse.sizeX == (13 + factor - 1) / factor && coarse.sizeY == (7 + factor - 1) / factor &&
               coarse.sizeZ == (10 + factor - 1) / factor);
        EXPECT(Near(coarse.voxelSize, 0.1 * factor, 1e-6, "coarse voxel size"));
        EXPECT(coarse.origin.x == grid.origin.x && coarse.origin.y == grid.origin.y &&
               coarse.origin.z == grid.origin.z);
        double coarseMass = 0.0;
        for (float d : coarse.density)
        {
            coarseMass += d;
        }
        coarseMass *= std::pow(double(coarse.voxelSize), 3.0);
        EXPECT(Near(coarseMass, mass, 1e-4 * mass, "mass"));
    }
    // BuildVolume averages a grid larger than maxResolution down to it.
    const renderer::VolumeResources limited = renderer::BuildVolume({.enabled = true, .maxResolution = 32}, grid);
    EXPECT(limited.data.sizeX == 13u); // within the limit: as it is
    const renderer::VolumeResources halved = renderer::BuildVolume({.enabled = true, .maxResolution = 8}, grid);
    EXPECT(halved.data.sizeX == 7u && halved.data.sizeY == 4u && halved.data.sizeZ == 5u);
}

TEST(Volume_PhaseSamplingFollowsThePhase)
{
    const renderer::VolumeResources none;
    VolumeReader reader{none};
    for (float g : {-0.5f, 0.0f, 0.35f, 0.9f})
    {
        VolumeData v{};
        v.anisotropy = g;
        // Normalized over the sphere.
        constexpr int kSteps = 200000;
        double integral = 0.0;
        for (int i = 0; i < kSteps; ++i)
        {
            const double u = -1.0 + 2.0 * (i + 0.5) / kSteps;
            integral += 2.0 * std::numbers::pi * reader.VolumePhase(v, float(u)) * 2.0 / kSteps;
        }
        EXPECT(Near(integral, 1.0, 2e-3, "phase integral over the sphere"));
        // The fraction of samples below each cosine is the CDF of Henyey-Greenstein there.
        const float3 axis = normalize(float3(-0.4f, 0.7f, 0.59f));
        const double cuts[] = {-0.9, -0.5, 0.0, 0.5, 0.9, 0.99};
        int below[std::size(cuts)] = {};
        constexpr int kSamples = 200000;
        uint seed = 9u;
        for (int i = 0; i < kSamples; ++i)
        {
            const float ux = VolumeReader::Rand(seed);
            const float2 u(ux, VolumeReader::Rand(seed));
            const float3 w = reader.SampleVolumePhase(v, axis, u);
            EXPECT(std::abs(length(w) - 1.0f) < 1e-4f);
            const double cosine = dot(w, axis);
            for (size_t k = 0; k < std::size(cuts); ++k)
            {
                below[k] += cosine <= cuts[k] ? 1 : 0;
            }
        }
        for (size_t k = 0; k < std::size(cuts); ++k)
        {
            // CDF of HG: (1 - g^2) / (2g) * (1 / sqrt(1 + g^2 - 2g u) - 1 / (1 + g)).
            const double c = cuts[k];
            const double cdf = std::abs(g) < 1e-6 ? (c + 1.0) / 2.0
                                                  : (1.0 - g * g) / (2.0 * g) *
                                                        (1.0 / std::sqrt(1.0 + g * g - 2.0 * g * c) - 1.0 / (1.0 + g));
            EXPECT(Near(double(below[k]) / kSamples, cdf, 5e-3, "sampled CDF"));
        }
    }
}

TEST(Volume_DisabledVolumeIsTransparent)
{
    const renderer::VolumeResources volume = renderer::BuildVolume({});
    EXPECT(volume.data.enabled == 0u && volume.density.empty() && volume.majorants.empty());
    VolumeReader reader{volume};
    uint seed = 1u;
    float tHit = 0.0f;
    EXPECT(reader.VolumeTransmittance(volume.data, float3(0.0f), float3(0.0f, 1.0f, 0.0f), 1e30f, seed) == 1.0f);
    EXPECT(!reader.VolumeCollision(volume.data, float3(0.0f), float3(0.0f, 1.0f, 0.0f), 1e30f, seed, tHit));
    // A file that cannot be read gives no volume and says why.
    const renderer::VolumeResources missing = renderer::BuildVolume({.enabled = true, .file = "does/not/exist.vdb"});
    EXPECT(missing.data.enabled == 0u && !missing.error.empty());
}
