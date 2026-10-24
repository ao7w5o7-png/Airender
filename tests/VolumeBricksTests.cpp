// A volume's bricks (renderer::BuildVolumeBricks, shader/VolumeBricks.hlsli): the 4-bit voxels read
// back the densities of the path tracer's texture to within their quantization, empty bricks hide no
// density and the regions a march skips are empty, the levels keep the mass, the bounds hold every
// density, and the ambient occlusion grid darkens towards the plume's core.

#include "TestFramework.h"

#include "air/renderer/BlueNoise.h"
#include "air/renderer/Clouds.h"
#include "air/renderer/TextureSampling.h"
#include "air/renderer/Transfer.h"
#include "air/renderer/Volume.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace air;
using namespace air::hlsl;
using namespace air::gpu;

namespace
{

// VolumeBricks.hlsli and VolumeLookup.hlsli (the path tracer's texture) over one volume in memory.
// The CPU tables are addressed by level: a level's buffer fields hold its index.
struct BrickReader
{
    const renderer::VolumeResources& volume;

    const renderer::VolumeBricks::Level& Level(VolumeBrickLevel l) const { return volume.bricks.levels[l.indirection]; }
    uint VolumeIndirectionLoad(VolumeBrickLevel l, uint index) const { return Level(l).indirection[index]; }
    uint VolumeOccupancyLoad(VolumeBrickLevel l, uint index) const { return Level(l).occupancy[index]; }
    uint VolumeBrickWordLoad(VolumeBrickLevel l, uint index) const { return Level(l).bricks[index]; }
    uint VolumeRangeLoad(VolumeBrickLevel l, uint brick) const { return Level(l).ranges[brick]; }
    float VolumeDensitySample(VolumeData v, float3 uvw) const
    {
        return renderer::SampleTexture3DClampedR16(volume.density.data(), v.sizeX + 2, v.sizeY + 2, v.sizeZ + 2, uvw.x,
                                                   uvw.y, uvw.z);
    }
    float VolumeMajorantLoad(VolumeData /*v*/, uint cell) const { return volume.majorants[cell]; }
    float CloudBaseSample(CloudData /*c*/, float3 /*uvw*/) const { return 0.0f; }
    float CloudDetailSample(CloudData /*c*/, float3 /*uvw*/) const { return 0.0f; }
    float CloudMajorantLoad(CloudData /*c*/, uint /*cell*/) const { return 0.0f; }

    VolumeBrickLevel LevelData(u32 level) const
    {
        VolumeBrickLevel l = volume.bricks.levels[level].data;
        l.indirection = level;
        return l;
    }
    static float Rand(uint& seed)
    {
        seed = Hash(seed);
        return float(seed >> 8) * (1.0f / 16777216.0f);
    }
    static uint Hash(uint v)
    {
        const uint state = v * 747796405u + 2891336453u;
        const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

// NOLINTBEGIN(readability-identifier-naming)
#include "air/renderer/shader/CloudLookup.hlsli"
#include "air/renderer/shader/VolumeBricks.hlsli"
#include "air/renderer/shader/VolumeLookup.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

const renderer::VolumeResources& Plume()
{
    static const renderer::VolumeResources volume =
        renderer::BuildVolume({.enabled = true, .position = {0.0f, 0.58f, 0.0f}, .extinction = 24.0f});
    return volume;
}

u32 Hash(u32 v)
{
    u32 state = v * 747796405u + 2891336453u;
    u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Uniform(u32& seed)
{
    seed = Hash(seed);
    return float(seed >> 8) * (1.0f / 16777216.0f);
}

// A point in the plume's padded box.
float3 RandomPoint(const VolumeData& v, u32& seed)
{
    const float3 lo = v.origin - float3(v.voxelSize);
    const float3 size = float3(float(v.sizeX + 2), float(v.sizeY + 2), float(v.sizeZ + 2)) * v.voxelSize;
    const float x = Uniform(seed);
    const float y = Uniform(seed);
    const float z = Uniform(seed);
    return lo + float3(x, y, z) * size;
}

} // namespace

// At level 0 the bricks hold the texture's voxels as 4 bits between each brick's lowest and highest
// density: a lookup differs from the texture's by at most 1/30 of its brick's range (plus the
// texture's 8-bit filter weights), and the optical depth along a ray by a fraction of a percent.
TEST(VolumeBricks_ReadTheVoxelsBack)
{
    const renderer::VolumeResources& volume = Plume();
    BrickReader reader{volume};
    const VolumeBrickLevel level = reader.LevelData(0);
    EXPECT(level.voxelsX == volume.data.sizeX + 2 && level.voxelsY == volume.data.sizeY + 2);
    EXPECT(level.brickCount > 0 && level.brickCount < level.bricksX * level.bricksY * level.bricksZ);
    u32 seed = 5;
    double sumError = 0.0;
    double worstRelative = 0.0; // of the brick's range
    int count = 0;
    for (int i = 0; i < 200000; ++i)
    {
        const float3 p = RandomPoint(volume.data, seed);
        const float texture = reader.VolumeDensity(volume.data, p);
        const float brick = reader.BrickDensity(level, p);
        sumError += std::abs(brick - texture);
        ++count;
        // The brick's range, to bound the quantization.
        const float fx = std::floor((p.x - level.corner.x) / level.voxelSize - 0.5f);
        const float fy = std::floor((p.y - level.corner.y) / level.voxelSize - 0.5f);
        const float fz = std::floor((p.z - level.corner.z) / level.voxelSize - 0.5f);
        if (fx < 0.0f || fy < 0.0f || fz < 0.0f || fx >= float(level.voxelsX - 1) || fy >= float(level.voxelsY - 1) ||
            fz >= float(level.voxelsZ - 1))
        {
            EXPECT(brick == 0.0f);
            continue;
        }
        const u32 b = reader.VolumeIndirectionLoad(
            level, ((u32(fz) >> 2) * level.bricksY + (u32(fy) >> 2)) * level.bricksX + (u32(fx) >> 2));
        if (b == kNoBrick)
        {
            EXPECT(brick == 0.0f && texture < 1e-6f);
            continue;
        }
        const u32 range = reader.VolumeRangeLoad(level, b);
        const float span = f16tof32(range >> 16) - f16tof32(range & 0xFFFFu);
        if (span > 1e-3f)
        {
            worstRelative = std::max(worstRelative, double(std::abs(brick - texture) / span));
        }
    }
    const double mean = sumError / count;
    std::printf("  level 0: %u bricks of %u, mean |difference| %.5f, worst %.3f of a brick's range\n", level.brickCount,
                level.bricksX * level.bricksY * level.bricksZ, mean, worstRelative);
    EXPECT(mean < 2e-3);
    EXPECT(worstRelative < 1.0 / 30.0 + 0.01);
    // Optical depth along random rays through the plume, marched at a tenth of a voxel: the errors of
    // the voxels average out along a ray, least along those that only graze the plume's thin fringe,
    // where a brick's 16 steps are coarse next to the faint densities.
    double worstDepth = 0.0;
    double worstThick = 0.0; // rays whose optical depth exceeds 2.4 (0.1 of density over a metre at 24 /m)
    double sumDepth = 0.0;
    int rays = 0;
    for (int ray = 0; ray < 400; ++ray)
    {
        const float3 o = RandomPoint(volume.data, seed);
        const float3 target(0.0f, 0.58f + 1.6f * Uniform(seed), 0.0f);
        const float3 d = normalize(target - o);
        double depthTexture = 0.0;
        double depthBricks = 0.0;
        const float step = volume.data.voxelSize * 0.1f;
        for (float t = 0.5f * step; t < 4.0f; t += step)
        {
            depthTexture += reader.VolumeDensity(volume.data, o + d * t) * step;
            depthBricks += reader.BrickDensity(level, o + d * t) * step;
        }
        if (depthTexture > 0.01)
        {
            const double error = std::abs(depthBricks / depthTexture - 1.0);
            worstDepth = std::max(worstDepth, error);
            worstThick = depthTexture > 0.1 ? std::max(worstThick, error) : worstThick;
            sumDepth += error;
            ++rays;
        }
    }
    std::printf("  optical depth along %d rays: %.3f%% apart on average, worst %.3f%% (%.3f%% where it exceeds 2.4)\n",
                rays, 100.0 * sumDepth / rays, worstDepth * 100.0, worstThick * 100.0);
    EXPECT(sumDepth / rays < 0.005);
    EXPECT(worstDepth < 0.05);
    EXPECT(worstThick < 0.02);
}

// Where BrickOccupied says a lookup reads zero, the texture reads zero, and so do lookups anywhere in
// the region it reports (what a march skips); every density above zero lies within the bounds.
TEST(VolumeBricks_SkippedRegionsAreEmpty)
{
    const renderer::VolumeResources& volume = Plume();
    BrickReader reader{volume};
    const VolumeMarchData& march = volume.bricks.data;
    u32 seed = 9;
    int skipped = 0;
    for (u32 levelIndex = 0; levelIndex < kVolumeLevels; ++levelIndex)
    {
        const VolumeBrickLevel level = reader.LevelData(levelIndex);
        for (int i = 0; i < 50000; ++i)
        {
            const float3 p = RandomPoint(volume.data, seed);
            float3 lo;
            float3 hi;
            const bool occupied = reader.BrickOccupied(level, p, lo, hi);
            const float density = reader.BrickDensity(level, p);
            if (density > 0.0f)
            {
                EXPECT(occupied);
                EXPECT(p.x >= march.boundsMin.x && p.y >= march.boundsMin.y && p.z >= march.boundsMin.z &&
                       p.x <= march.boundsMax.x && p.y <= march.boundsMax.y && p.z <= march.boundsMax.z);
            }
            if (occupied || lo == hi)
            {
                continue;
            }
            ++skipped;
            EXPECT(p.x >= lo.x && p.y >= lo.y && p.z >= lo.z && p.x <= hi.x && p.y <= hi.y && p.z <= hi.z);
            for (int k = 0; k < 4; ++k)
            {
                const float u = Uniform(seed);
                const float v = Uniform(seed);
                const float w = Uniform(seed);
                const float3 q = lo + (hi - lo) * float3(u, v, w);
                EXPECT(reader.BrickDensity(level, q) == 0.0f);
                if (levelIndex == 0)
                {
                    EXPECT(reader.VolumeDensity(volume.data, q) < 1e-6f);
                }
            }
        }
    }
    EXPECT(skipped > 1000);
}

// Each level averages the one below over 2^3 voxels: the same mass, half the resolution, far fewer
// bricks. The ambient occlusion grid lies in [0, 1], is 1 where there is no smoke and darkest in it.
TEST(VolumeBricks_LevelsKeepTheMassAndOcclusionDarkensTheCore)
{
    const renderer::VolumeResources& volume = Plume();
    BrickReader reader{volume};
    double mass[kVolumeLevels] = {};
    for (u32 levelIndex = 0; levelIndex < kVolumeLevels; ++levelIndex)
    {
        const VolumeBrickLevel level = reader.LevelData(levelIndex);
        EXPECT(std::abs(level.voxelSize - volume.data.voxelSize * float(1u << levelIndex)) < 1e-7f);
        // The integral of the density over the box, on a fine lattice.
        const float3 lo = volume.data.origin;
        const float3 size = float3(float(volume.data.sizeX), float(volume.data.sizeY), float(volume.data.sizeZ)) *
                            volume.data.voxelSize;
        constexpr int kSamples = 48;
        for (int z = 0; z < kSamples; ++z)
        {
            for (int y = 0; y < 2 * kSamples; ++y)
            {
                for (int x = 0; x < kSamples; ++x)
                {
                    const float3 p =
                        lo + size * float3((float(x) + 0.5f) / kSamples, (float(y) + 0.5f) / (2 * kSamples),
                                           (float(z) + 0.5f) / kSamples);
                    mass[levelIndex] += reader.BrickDensity(level, p);
                }
            }
        }
        std::printf("  level %u: %u bricks, mass %.1f\n", levelIndex, level.brickCount, mass[levelIndex]);
        if (levelIndex > 0)
        {
            EXPECT(level.brickCount < volume.bricks.levels[levelIndex - 1].data.brickCount);
            EXPECT(std::abs(mass[levelIndex] / mass[0] - 1.0) < 0.03);
        }
    }
    const VolumeMarchData& m = volume.bricks.data;
    EXPECT(volume.bricks.ao.size() == size_t(m.aoX) * m.aoY * m.aoZ);
    u8 darkest = 255;
    for (u8 texel : volume.bricks.ao)
    {
        darkest = std::min(darkest, texel);
    }
    // The corners of the box hold no smoke; the column's core does.
    EXPECT(volume.bricks.ao[0] > 200);
    EXPECT(darkest < 128);
    const u32 core = (m.aoZ / 2 * m.aoY + m.aoY / 3) * m.aoX + m.aoX / 2;
    EXPECT(volume.bricks.ao[core] < volume.bricks.ao[0]);
}

// The blue noise is a permutation of its ranks without low frequencies: its spectrum near zero holds
// a small part of the power white noise would.
TEST(BlueNoise_IsAPermutationWithoutLowFrequencies)
{
    const std::vector<u16>& noise = renderer::BlueNoise();
    constexpr u32 kSize = renderer::kBlueNoiseSize;
    EXPECT(noise.size() == size_t(kSize) * kSize);
    std::vector<bool> seen(noise.size(), false);
    for (u16 rank : noise)
    {
        EXPECT(rank < noise.size() && !seen[rank]);
        seen[rank] = true;
    }
    // Power at frequencies 1 to 4 (radius), against the expectation for white noise of the same
    // variance (size^2 * variance per frequency).
    double low = 0.0;
    int frequencies = 0;
    const double n = double(noise.size());
    for (int ky = -4; ky <= 4; ++ky)
    {
        for (int kx = -4; kx <= 4; ++kx)
        {
            if ((kx == 0 && ky == 0) || kx * kx + ky * ky > 16)
            {
                continue;
            }
            std::complex<double> sum = 0.0;
            for (u32 y = 0; y < kSize; ++y)
            {
                for (u32 x = 0; x < kSize; ++x)
                {
                    const double value = (double(noise[y * kSize + x]) + 0.5) / n - 0.5;
                    const double phase = -6.283185307179586 * (double(kx) * x + double(ky) * y) / kSize;
                    sum += value * std::complex<double>(std::cos(phase), std::sin(phase));
                }
            }
            low += std::norm(sum);
            ++frequencies;
        }
    }
    const double white = n * (1.0 / 12.0);
    std::printf("  low-frequency power: %.4f of white noise's\n", low / frequencies / white);
    EXPECT(low / frequencies < 0.05 * white);
}
