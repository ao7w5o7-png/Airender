// The bricks behind BuildVolumeBricks (shader/VolumeBricksShared.h, docs/architecture.md §8.13).

#include "air/renderer/Transfer.h"
#include "air/renderer/Volume.h"

#include "ParallelFor.h"

#include <algorithm>
#include <cmath>

namespace air::renderer
{

using hlsl::float3;

namespace
{

// A grid's voxels with a zero voxel on every side: (x, y, z) in padded coordinates.
float PaddedAt(const VolumeGrid& grid, int x, int y, int z)
{
    if (x < 1 || y < 1 || z < 1 || x > int(grid.sizeX) || y > int(grid.sizeY) || z > int(grid.sizeZ))
    {
        return 0.0f;
    }
    return std::max(grid.At(u32(x - 1), u32(y - 1), u32(z - 1)), 0.0f);
}

struct Encoded
{
    bool allocated = false;
    u32 range = 0;
    u32 words[gpu::kBrickWords] = {};
};

// Brick (bx, by, bz): its 125 voxels as 4-bit codes between its lowest and highest density, those
// rounded down and up to half floats so that every voxel lies between them.
Encoded EncodeBrick(const VolumeGrid& grid, u32 bx, u32 by, u32 bz)
{
    Encoded out;
    float values[gpu::kBrickStored * gpu::kBrickStored * gpu::kBrickStored];
    float lowest = 1e30f;
    float highest = 0.0f;
    u32 v = 0;
    for (u32 z = 0; z < gpu::kBrickStored; ++z)
    {
        for (u32 y = 0; y < gpu::kBrickStored; ++y)
        {
            for (u32 x = 0; x < gpu::kBrickStored; ++x, ++v)
            {
                values[v] = PaddedAt(grid, int(bx * gpu::kBrickVoxels + x), int(by * gpu::kBrickVoxels + y),
                                     int(bz * gpu::kBrickVoxels + z));
                lowest = std::min(lowest, values[v]);
                highest = std::max(highest, values[v]);
            }
        }
    }
    if (!(highest > 0.0f))
    {
        return out;
    }
    out.allocated = true;
    u16 low = FloatToHalf(lowest);
    if (HalfToFloat(low) > lowest)
    {
        --low; // densities are not negative: one step down in the encoding is one half float lower
    }
    u16 high = FloatToHalf(std::min(highest, 65504.0f)); // the largest half float
    if (HalfToFloat(high) < highest && high < 0x7BFFu)
    {
        ++high;
    }
    out.range = u32(low) | (u32(high) << 16);
    const float lo = HalfToFloat(low);
    const float span = HalfToFloat(high) - lo;
    for (v = 0; v < gpu::kBrickStored * gpu::kBrickStored * gpu::kBrickStored; ++v)
    {
        const u32 code = span > 0.0f ? u32(std::clamp(std::round((values[v] - lo) / span * 15.0f), 0.0f, 15.0f)) : 0u;
        out.words[v >> 3] |= code << ((v & 7u) * 4u);
    }
    return out;
}

VolumeBricks::Level BuildLevel(const VolumeGrid& grid, float3 origin, float voxelSize)
{
    VolumeBricks::Level out;
    gpu::VolumeBrickLevel& l = out.data;
    l.corner = origin - float3(voxelSize);
    l.voxelSize = voxelSize;
    l.voxelsX = grid.sizeX + 2;
    l.voxelsY = grid.sizeY + 2;
    l.voxelsZ = grid.sizeZ + 2;
    l.bricksX = (l.voxelsX + gpu::kBrickVoxels - 1) / gpu::kBrickVoxels;
    l.bricksY = (l.voxelsY + gpu::kBrickVoxels - 1) / gpu::kBrickVoxels;
    l.bricksZ = (l.voxelsZ + gpu::kBrickVoxels - 1) / gpu::kBrickVoxels;
    l.wordsX = (l.bricksX + gpu::kOccupancyX - 1) / gpu::kOccupancyX;
    l.wordsY = (l.bricksY + gpu::kOccupancyY - 1) / gpu::kOccupancyY;
    l.wordsZ = (l.bricksZ + gpu::kOccupancyZ - 1) / gpu::kOccupancyZ;
    const size_t positions = size_t(l.bricksX) * l.bricksY * l.bricksZ;
    std::vector<Encoded> encoded(positions);
    detail::ParallelFor(int(l.bricksZ),
                        [&](int bz)
                        {
                            for (u32 by = 0; by < l.bricksY; ++by)
                            {
                                for (u32 bx = 0; bx < l.bricksX; ++bx)
                                {
                                    encoded[(size_t(bz) * l.bricksY + by) * l.bricksX + bx] =
                                        EncodeBrick(grid, bx, by, u32(bz));
                                }
                            }
                        });
    // Bricks in the order of their positions, so the layout does not depend on the threads.
    out.indirection.assign(positions, gpu::kNoBrick);
    out.occupancy.assign(size_t(l.wordsX) * l.wordsY * l.wordsZ, 0u);
    for (u32 bz = 0; bz < l.bricksZ; ++bz)
    {
        for (u32 by = 0; by < l.bricksY; ++by)
        {
            for (u32 bx = 0; bx < l.bricksX; ++bx)
            {
                const size_t position = (size_t(bz) * l.bricksY + by) * l.bricksX + bx;
                const Encoded& e = encoded[position];
                if (!e.allocated)
                {
                    continue;
                }
                out.indirection[position] = u32(out.ranges.size());
                out.ranges.push_back(e.range);
                out.bricks.insert(out.bricks.end(), std::begin(e.words), std::end(e.words));
                const size_t word = (size_t(bz / gpu::kOccupancyZ) * l.wordsY + by / gpu::kOccupancyY) * l.wordsX +
                                    bx / gpu::kOccupancyX;
                out.occupancy[word] |= 1u << ((bx % gpu::kOccupancyX) + gpu::kOccupancyX * (by % gpu::kOccupancyY) +
                                              gpu::kOccupancyX * gpu::kOccupancyY * (bz % gpu::kOccupancyZ));
            }
        }
    }
    l.brickCount = u32(out.ranges.size());
    return out;
}

// Grows [lo, hi] by the lookups of a level that can read a density above zero: those whose lowest
// voxel lies in an allocated brick, [4 b + 0.5, 4 b + 4.5) voxels from the corner, within the
// voxels whose lookups can read anything at all.
void GrowBounds(const VolumeBricks::Level& level, float3& lo, float3& hi)
{
    const gpu::VolumeBrickLevel& l = level.data;
    u32 b0[3] = {~0u, ~0u, ~0u};
    u32 b1[3] = {0, 0, 0};
    for (u32 bz = 0; bz < l.bricksZ; ++bz)
    {
        for (u32 by = 0; by < l.bricksY; ++by)
        {
            for (u32 bx = 0; bx < l.bricksX; ++bx)
            {
                if (level.indirection[(size_t(bz) * l.bricksY + by) * l.bricksX + bx] == gpu::kNoBrick)
                {
                    continue;
                }
                const u32 b[3] = {bx, by, bz};
                for (int a = 0; a < 3; ++a)
                {
                    b0[a] = std::min(b0[a], b[a]);
                    b1[a] = std::max(b1[a], b[a]);
                }
            }
        }
    }
    if (b0[0] == ~0u)
    {
        return;
    }
    const u32 voxels[3] = {l.voxelsX, l.voxelsY, l.voxelsZ};
    float from[3];
    float to[3];
    for (int a = 0; a < 3; ++a)
    {
        from[a] = std::max(float(b0[a] * gpu::kBrickVoxels) + 0.5f, 0.5f);
        to[a] = std::min(float((b1[a] + 1) * gpu::kBrickVoxels) + 0.5f, float(voxels[a]) - 0.5f);
    }
    lo = min(lo, l.corner + float3(from[0], from[1], from[2]) * l.voxelSize);
    hi = max(hi, l.corner + float3(to[0], to[1], to[2]) * l.voxelSize);
}

// Trilinear lookup in a grid's padded voxels, (x, y, z) in voxels from the padded corner.
float PaddedTrilinear(const VolumeGrid& grid, float x, float y, float z)
{
    const float qx = x - 0.5f;
    const float qy = y - 0.5f;
    const float qz = z - 0.5f;
    const int ix = int(std::floor(qx));
    const int iy = int(std::floor(qy));
    const int iz = int(std::floor(qz));
    const float tx = qx - float(ix);
    const float ty = qy - float(iy);
    const float tz = qz - float(iz);
    auto row = [&](int j, int k)
    { return PaddedAt(grid, ix, iy + j, iz + k) * (1.0f - tx) + PaddedAt(grid, ix + 1, iy + j, iz + k) * tx; };
    const float r0 = row(0, 0) * (1.0f - ty) + row(1, 0) * ty;
    const float r1 = row(0, 1) * (1.0f - ty) + row(1, 1) * ty;
    return r0 * (1.0f - tz) + r1 * tz;
}

} // namespace

VolumeBricks BuildVolumeBricks(const Volume& volume, const VolumeGrid& grid, float3 origin, float voxelSize)
{
    VolumeBricks out;
    gpu::VolumeMarchData& d = out.data;
    d.extinction = volume.extinction;
    d.albedo = std::clamp(volume.albedo, 0.0f, 1.0f);
    d.anisotropy = std::clamp(volume.anisotropy, -0.99f, 0.99f);
    // Level l averages blocks of 2^l voxels; every level shares the voxels' corner.
    VolumeGrid coarse[gpu::kVolumeLevels];
    float3 lo(1e30f);
    float3 hi(-1e30f);
    for (u32 level = 0; level < gpu::kVolumeLevels; ++level)
    {
        if (level > 0)
        {
            coarse[level] = Downsample(level == 1 ? grid : coarse[level - 1], 2);
        }
        const VolumeGrid& source = level == 0 ? grid : coarse[level];
        out.levels[level] = BuildLevel(source, origin, voxelSize * float(1u << level));
        GrowBounds(out.levels[level], lo, hi);
    }
    d.boundsMin = lo;
    d.boundsMax = hi;

    // The volume's own mean transmittance over all directions, from the centre of every texel of
    // kVolumeAoCell level 0 voxels: the coarsest level marched voxel by voxel along 32 directions
    // spread over the sphere (a Fibonacci lattice).
    const gpu::VolumeBrickLevel& fine = out.levels[0].data;
    d.aoCell = float(gpu::kVolumeAoCell) * voxelSize;
    d.aoCorner = fine.corner;
    d.aoX = (fine.voxelsX + gpu::kVolumeAoCell - 1) / gpu::kVolumeAoCell;
    d.aoY = (fine.voxelsY + gpu::kVolumeAoCell - 1) / gpu::kVolumeAoCell;
    d.aoZ = (fine.voxelsZ + gpu::kVolumeAoCell - 1) / gpu::kVolumeAoCell;
    out.ao.assign(size_t(d.aoX) * d.aoY * d.aoZ, 255);
    const u32 last = gpu::kVolumeLevels - 1;
    const VolumeGrid& march = last == 0 ? grid : coarse[last];
    const gpu::VolumeBrickLevel& marchLevel = out.levels[last].data;
    constexpr u32 kDirections = 32;
    float3 directions[kDirections];
    for (u32 i = 0; i < kDirections; ++i)
    {
        const float z = 1.0f - (2.0f * float(i) + 1.0f) / float(kDirections);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = 2.39996323f * float(i); // the golden angle
        directions[i] = float3(r * std::cos(phi), r * std::sin(phi), z);
    }
    const float3 extent = float3(float(marchLevel.voxelsX), float(marchLevel.voxelsY), float(marchLevel.voxelsZ));
    detail::ParallelFor(
        int(d.aoZ),
        [&](int z)
        {
            for (u32 y = 0; y < d.aoY; ++y)
            {
                for (u32 x = 0; x < d.aoX; ++x)
                {
                    const float3 centre =
                        d.aoCorner + float3(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f) * d.aoCell;
                    // In the coarse level's voxels from its padded corner.
                    const float3 start = (centre - marchLevel.corner) / marchLevel.voxelSize;
                    double sum = 0.0;
                    for (const float3& dir : directions)
                    {
                        float depth = 0.0f;
                        for (float t = 0.5f;; t += 1.0f)
                        {
                            const float3 p = start + dir * t;
                            if (p.x < 0.0f || p.y < 0.0f || p.z < 0.0f || p.x > extent.x || p.y > extent.y ||
                                p.z > extent.z)
                            {
                                break;
                            }
                            depth += PaddedTrilinear(march, p.x, p.y, p.z);
                        }
                        sum += std::exp(-double(depth) * volume.extinction * marchLevel.voxelSize);
                    }
                    const double mean = sum / double(kDirections);
                    out.ao[(size_t(z) * d.aoY + y) * d.aoX + x] = u8(std::lround(std::clamp(mean, 0.0, 1.0) * 255.0));
                }
            }
        });
    return out;
}

} // namespace air::renderer
