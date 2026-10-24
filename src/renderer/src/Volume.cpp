#include "air/renderer/Volume.h"

#include "air/core/Log.h"
#include "air/renderer/Transfer.h"
#include "air/renderer/Vdb.h"

#include "ParallelFor.h"

#include <algorithm>
#include <cmath>

namespace air::renderer
{

using hlsl::float3;

namespace
{

u32 Hash(u32 x, u32 y, u32 z, u32 seed)
{
    u32 h = x * 0x8da6b343u ^ y * 0xd8163841u ^ z * 0xcb1ab31fu ^ seed * 0x165667b1u;
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    h *= 0x297a2d39u;
    h ^= h >> 15;
    return h;
}

float Lattice(int x, int y, int z, u32 seed)
{
    return float(Hash(u32(x), u32(y), u32(z), seed) & 0xFFFFFFu) / 16777215.0f;
}

float Smooth(float t)
{
    return t * t * (3.0f - 2.0f * t);
}

// Value noise in [0, 1], smooth between integer lattice points.
float ValueNoise(float x, float y, float z, u32 seed)
{
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const float fz = std::floor(z);
    const int ix = int(fx);
    const int iy = int(fy);
    const int iz = int(fz);
    const float tx = Smooth(x - fx);
    const float ty = Smooth(y - fy);
    const float tz = Smooth(z - fz);
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    auto row = [&](int j, int k)
    { return lerp(Lattice(ix, iy + j, iz + k, seed), Lattice(ix + 1, iy + j, iz + k, seed), tx); };
    return lerp(lerp(row(0, 0), row(1, 0), ty), lerp(row(0, 1), row(1, 1), ty), tz);
}

// Four octaves of value noise, in [0, 1].
float Fbm(float x, float y, float z, u32 seed)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float total = 0.0f;
    for (u32 octave = 0; octave < 4; ++octave)
    {
        sum += amplitude * ValueNoise(x, y, z, seed + octave * 101u);
        total += amplitude;
        x *= 2.03f;
        y *= 2.03f;
        z *= 2.03f;
        amplitude *= 0.5f;
    }
    return sum / total;
}

float Smoothstep(float a, float b, float x)
{
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

} // namespace

VolumeGrid MakeSmokePlume(u32 seed)
{
    VolumeGrid grid;
    grid.sizeX = 64;
    grid.sizeY = 128;
    grid.sizeZ = 64;
    grid.voxelSize = 0.025f;
    grid.origin = float3(-0.5f * float(grid.sizeX) * grid.voxelSize, 0.0f, -0.5f * float(grid.sizeZ) * grid.voxelSize);
    grid.density.assign(size_t(grid.sizeX) * grid.sizeY * grid.sizeZ, 0.0f);
    const float phase = float(Hash(seed, 1, 2, 3) & 0xFFFFu) / 65535.0f * 6.2831853f;
    detail::ParallelFor(int(grid.sizeY),
                        [&](int y)
                        {
                            // h runs up the column (0..1); x and z across it, in units of its width.
                            const float h = (float(y) + 0.5f) / float(grid.sizeY);
                            const float swayX = 0.12f * h * std::sin(h * 7.0f + phase);
                            const float swayZ = 0.12f * h * std::cos(h * 5.0f + phase * 1.7f);
                            const float radius = 0.05f + 0.3f * std::pow(h, 0.8f);
                            const float fade = Smoothstep(0.0f, 0.04f, h) * (1.0f - Smoothstep(0.7f, 1.0f, h));
                            for (u32 z = 0; z < grid.sizeZ; ++z)
                            {
                                for (u32 x = 0; x < grid.sizeX; ++x)
                                {
                                    const float px = (float(x) + 0.5f) / float(grid.sizeX) - 0.5f;
                                    const float pz = (float(z) + 0.5f) / float(grid.sizeZ) - 0.5f;
                                    const float r =
                                        std::sqrt((px - swayX) * (px - swayX) + (pz - swayZ) * (pz - swayZ));
                                    const float turbulence = Fbm(px * 6.0f, h * 12.0f, pz * 6.0f, seed);
                                    const float edge = radius * (0.65f + 0.7f * turbulence);
                                    const float body = 1.0f - Smoothstep(edge * 0.5f, edge, r);
                                    const float puffs =
                                        0.35f + 0.65f * Fbm(px * 14.0f, h * 28.0f, pz * 14.0f, seed + 7u);
                                    grid.density[(size_t(z) * grid.sizeY + size_t(y)) * grid.sizeX + x] =
                                        body * fade * puffs * (1.0f - 0.5f * h);
                                }
                            }
                        });
    return grid;
}

VolumeGrid Downsample(const VolumeGrid& grid, u32 factor)
{
    if (factor <= 1)
    {
        return grid;
    }
    VolumeGrid out;
    out.sizeX = (grid.sizeX + factor - 1) / factor;
    out.sizeY = (grid.sizeY + factor - 1) / factor;
    out.sizeZ = (grid.sizeZ + factor - 1) / factor;
    out.voxelSize = grid.voxelSize * float(factor);
    out.origin = grid.origin;
    out.density.assign(size_t(out.sizeX) * out.sizeY * out.sizeZ, 0.0f);
    detail::ParallelFor(int(out.sizeZ),
                        [&](int z)
                        {
                            for (u32 y = 0; y < out.sizeY; ++y)
                            {
                                for (u32 x = 0; x < out.sizeX; ++x)
                                {
                                    // The voxels outside the grid count as empty, like the space
                                    // around it.
                                    double sum = 0.0;
                                    for (u32 k = 0; k < factor; ++k)
                                    {
                                        for (u32 j = 0; j < factor; ++j)
                                        {
                                            for (u32 i = 0; i < factor; ++i)
                                            {
                                                const u32 sx = x * factor + i;
                                                const u32 sy = y * factor + j;
                                                const u32 sz = u32(z) * factor + k;
                                                if (sx < grid.sizeX && sy < grid.sizeY && sz < grid.sizeZ)
                                                {
                                                    sum += grid.At(sx, sy, sz);
                                                }
                                            }
                                        }
                                    }
                                    out.density[(size_t(z) * out.sizeY + y) * out.sizeX + x] =
                                        float(sum / double(factor * factor * factor));
                                }
                            }
                        });
    return out;
}

std::optional<VolumeGrid> LoadVolumeGrid(const Volume& volume, std::string* error)
{
    if (volume.file.empty())
    {
        return MakeSmokePlume(volume.seed);
    }
    return ReadVdbGrid(volume.file, volume.grid, std::max(volume.maxResolution, 8u), error);
}

VolumeResources BuildVolume(const Volume& volume)
{
    VolumeResources none;
    if (!volume.enabled)
    {
        return none;
    }
    std::string error;
    const std::optional<VolumeGrid> grid = LoadVolumeGrid(volume, &error);
    if (!grid)
    {
        none.error = error;
        AIR_LOG(Error, "Volume", "{}", error);
        return none;
    }
    return BuildVolume(volume, *grid);
}

VolumeResources BuildVolume(const Volume& volume, const VolumeGrid& source)
{
    VolumeResources out;
    if (!volume.enabled || source.density.empty())
    {
        return out;
    }
    const u32 longest = std::max(std::max(source.sizeX, source.sizeY), source.sizeZ);
    const u32 factor = (longest + std::max(volume.maxResolution, 8u) - 1) / std::max(volume.maxResolution, 8u);
    const VolumeGrid grid = Downsample(source, factor);

    // Placement.
    float voxelSize = grid.voxelSize;
    float3 origin = grid.origin + volume.position;
    if (volume.size > 0.0f)
    {
        voxelSize = volume.size / float(std::max(std::max(grid.sizeX, grid.sizeY), grid.sizeZ));
        origin =
            volume.position - float3(0.5f * float(grid.sizeX) * voxelSize, 0.0f, 0.5f * float(grid.sizeZ) * voxelSize);
    }

    // Half-float texels with a zero voxel on every side.
    const u32 px = grid.sizeX + 2;
    const u32 py = grid.sizeY + 2;
    const u32 pz = grid.sizeZ + 2;
    out.density.assign(size_t(px) * py * pz, FloatToHalf(0.0f));
    for (u32 z = 0; z < grid.sizeZ; ++z)
    {
        for (u32 y = 0; y < grid.sizeY; ++y)
        {
            for (u32 x = 0; x < grid.sizeX; ++x)
            {
                out.density[(size_t(z + 1) * py + (y + 1)) * px + (x + 1)] =
                    FloatToHalf(std::max(grid.At(x, y, z), 0.0f));
            }
        }
    }

    // Majorants: the largest texel a trilinear lookup inside a cell can read (one texel beyond it
    // on every side), as rounded to half floats.
    gpu::VolumeData& d = out.data;
    d.origin = origin;
    d.voxelSize = voxelSize;
    d.sizeX = grid.sizeX;
    d.sizeY = grid.sizeY;
    d.sizeZ = grid.sizeZ;
    d.enabled = 1;
    d.extinction = volume.extinction;
    d.albedo = std::clamp(volume.albedo, 0.0f, 1.0f);
    d.anisotropy = std::clamp(volume.anisotropy, -0.99f, 0.99f);
    d.cellsX = (px + gpu::kVolumeCell - 1) / gpu::kVolumeCell;
    d.cellsY = (py + gpu::kVolumeCell - 1) / gpu::kVolumeCell;
    d.cellsZ = (pz + gpu::kVolumeCell - 1) / gpu::kVolumeCell;
    out.majorants.assign(size_t(d.cellsX) * d.cellsY * d.cellsZ, 0.0f);
    for (u32 cz = 0; cz < d.cellsZ; ++cz)
    {
        for (u32 cy = 0; cy < d.cellsY; ++cy)
        {
            for (u32 cx = 0; cx < d.cellsX; ++cx)
            {
                float largest = 0.0f;
                auto range = [](u32 cell, u32 size, int& lo, int& hi)
                {
                    lo = std::max(int(cell * gpu::kVolumeCell) - 1, 0);
                    hi = std::min(int((cell + 1) * gpu::kVolumeCell), int(size) - 1);
                };
                int x0, x1, y0, y1, z0, z1;
                range(cx, px, x0, x1);
                range(cy, py, y0, y1);
                range(cz, pz, z0, z1);
                for (int z = z0; z <= z1; ++z)
                {
                    for (int y = y0; y <= y1; ++y)
                    {
                        for (int x = x0; x <= x1; ++x)
                        {
                            largest = std::max(largest,
                                               HalfToFloat(out.density[(size_t(z) * py + size_t(y)) * px + size_t(x)]));
                        }
                    }
                }
                out.majorants[(size_t(cz) * d.cellsY + cy) * d.cellsX + cx] = largest * volume.extinction * 1.00001f;
            }
        }
    }
    return out;
}

} // namespace air::renderer
