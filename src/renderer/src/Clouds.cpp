// The procedural cumulus behind BuildCloudVolume (docs/architecture.md §8.10).
//
// A 2D weather field (gradient noise for regional variation plus cellular noise that gathers the
// clouds into cells) decides where clouds are: its quantile at 1 - coverage is the threshold, so the
// requested fraction of the tile lies under cloud bases. How far the weather exceeds the threshold
// sets how tall the cloud grows and how dense it is. Vertically each cloud has a flat base at the
// condensation level and a rounded top; billowy cellular noise carves its outline. A second, small
// periodic grid of finer cellular noise erodes the fringe at render time (CloudLookup.hlsli), which
// gives the outlines detail far below the base voxels' size, and the eroded density is sharpened
// into crisp outlines over even interiors. Everything is deterministic in the seed.

#include "air/renderer/Clouds.h"

#include "ParallelFor.h"

#include "air/core/Assert.h"

#include <algorithm>
#include <cmath>

namespace air::renderer
{
namespace
{

using detail::ParallelFor;

constexpr u32 kGridXZ = 256;          // base voxels across the tile
constexpr float kVoxel = 50.0f;       // their width at scale 1, metres: a 12.8 km tile
constexpr float kVoxelHeight = 25.0f; // their height at scale 1, about
constexpr u32 kCellXZ = 8;            // base voxels per majorant cell
constexpr u32 kCellY = 4;
constexpr u32 kDetailGrid = 64;         // detail voxels along each axis
constexpr float kDetailPeriod = 400.0f; // metres at scale 1: detail voxels of 6.25 m
constexpr float kDetailStrength = 0.75f;
constexpr float kSharpness = 10.0f;

// "lowbias32" integer hash (Wellons).
u32 Hash(u32 x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

u32 Hash(int x, int y, int z, u32 seed)
{
    return Hash(u32(x) ^ Hash(u32(y) ^ Hash(u32(z) ^ Hash(seed))));
}

float Unit(u32 h)
{
    return float(h >> 8) * (1.0f / 16777216.0f);
}

int Wrap(int i, int period)
{
    const int m = i % period;
    return m < 0 ? m + period : m;
}

float Saturate(float v)
{
    return std::clamp(v, 0.0f, 1.0f);
}

float Smoothstep(float a, float b, float v)
{
    const float t = Saturate((v - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

float Fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// 2D gradient noise (about -0.7..0.7) on unit cells, repeating every `period` cells.
float Perlin(float x, float z, int period, u32 seed)
{
    const float fx = std::floor(x);
    const float fz = std::floor(z);
    const int x0 = int(fx);
    const int z0 = int(fz);
    const float tx = x - fx;
    const float tz = z - fz;
    float corner[2][2];
    for (int dz = 0; dz < 2; ++dz)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            const float angle = 6.28318530718f * Unit(Hash(Wrap(x0 + dx, period), 0, Wrap(z0 + dz, period), seed));
            corner[dz][dx] = std::cos(angle) * (tx - float(dx)) + std::sin(angle) * (tz - float(dz));
        }
    }
    const float u = Fade(tx);
    const float v = Fade(tz);
    const float a = corner[0][0] + (corner[0][1] - corner[0][0]) * u;
    const float b = corner[1][0] + (corner[1][1] - corner[1][0]) * u;
    return a + (b - a) * v;
}

// Distance (0..1) from a point to the nearest of one random point per unit cell, repeating every
// `period` cells in x and z, and every periodY cells in y (0: not in y).
float Worley(float x, float y, float z, int period, int periodY, u32 seed)
{
    const int cx = int(std::floor(x));
    const int cy = int(std::floor(y));
    const int cz = int(std::floor(z));
    float best = 8.0f;
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int ix = cx + dx;
                const int iy = cy + dy;
                const int iz = cz + dz;
                const u32 h = Hash(Wrap(ix, period), periodY > 0 ? Wrap(iy, periodY) : iy, Wrap(iz, period), seed);
                const float px = float(ix) + Unit(h) - x;
                const float py = float(iy) + Unit(Hash(h + 1u)) - y;
                const float pz = float(iz) + Unit(Hash(h + 2u)) - z;
                best = std::min(best, px * px + py * py + pz * pz);
            }
        }
    }
    return std::min(std::sqrt(best), 1.0f);
}

float Worley(float x, float z, int period, u32 seed)
{
    const int cx = int(std::floor(x));
    const int cz = int(std::floor(z));
    float best = 8.0f;
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int ix = cx + dx;
            const int iz = cz + dz;
            const u32 h = Hash(Wrap(ix, period), 0, Wrap(iz, period), seed);
            const float px = float(ix) + Unit(h) - x;
            const float pz = float(iz) + Unit(Hash(h + 1u)) - z;
            best = std::min(best, px * px + pz * pz);
        }
    }
    return std::min(std::sqrt(best), 1.0f);
}

// Where the clouds are, over the tile's (u, w) in [0, 1)^2: higher is cloudier.
float Weather(float u, float w, u32 seed)
{
    float regional = 0.0f;
    float amplitude = 0.5f;
    for (int octave = 0, period = 4; octave < 4; ++octave, period *= 2)
    {
        regional += amplitude * Perlin(u * float(period), w * float(period), period, seed + u32(octave));
        amplitude *= 0.5f;
    }
    const float cells = 1.0f - Worley(u * 8.0f, w * 8.0f, 8, seed + 17u);
    return regional + 0.6f * cells;
}

// Billows over the tile at (u, height in metres, w): cellular noise, high near its points.
float Billows(float u, float y, float w, float tileSize, u32 seed)
{
    float sum = 0.0f;
    const float weights[3] = {0.5f, 0.3f, 0.2f};
    for (int octave = 0, period = 24; octave < 3; ++octave, period *= 2)
    {
        const float cell = tileSize / float(period);
        sum += weights[octave] *
               (1.0f - Worley(u * float(period), y / cell, w * float(period), period, 0, seed + 101u + u32(octave)));
    }
    return sum;
}

u8 Texel(float value)
{
    return u8(std::lround(Saturate(value) * 255.0f));
}

} // namespace

float SampleTexture3D(const u8* texels, u32 width, u32 height, u32 depth, float u, float v, float w)
{
    // Texel coordinates in fixed point with 8 fractional bits, as the filtering hardware rounds them.
    int i[3][2];
    float t[3];
    const float coords[3] = {u, v, w};
    const u32 sizes[3] = {width, height, depth};
    for (int a = 0; a < 3; ++a)
    {
        const float fixed = std::round((coords[a] * float(sizes[a]) - 0.5f) * 256.0f);
        const float whole = std::floor(fixed / 256.0f);
        t[a] = (fixed - whole * 256.0f) / 256.0f;
        i[a][0] = Wrap(int(whole), int(sizes[a]));
        i[a][1] = Wrap(int(whole) + 1, int(sizes[a]));
    }
    auto at = [&](int x, int y, int z)
    { return float(texels[(size_t(i[2][z]) * height + size_t(i[1][y])) * width + size_t(i[0][x])]) / 255.0f; };
    auto row = [&](int y, int z) { return at(0, y, z) + (at(1, y, z) - at(0, y, z)) * t[0]; };
    auto slice = [&](int z) { return row(0, z) + (row(1, z) - row(0, z)) * t[1]; };
    return slice(0) + (slice(1) - slice(0)) * t[2];
}

CloudPhaseParams FitCloudPhase(float diameter)
{
    const double d = std::clamp(double(diameter), 5.0, 50.0);
    CloudPhaseParams p;
    p.gHg = float(std::exp(-0.0990567 / (d - 1.67154)));
    p.gDraine = float(std::exp(-2.20679 / (d + 3.91029) - 0.428934));
    p.alphaDraine = float(std::exp(3.62489 - 8.29288 / (d + 5.52825)));
    p.wDraine = float(std::exp(-0.599085 / (d - 0.641583) - 0.665888));
    return p;
}

CloudVolume BuildCloudVolume(const Clouds& clouds)
{
    CloudVolume volume;
    gpu::CloudData& data = volume.data;
    data.bottom = clouds.bottom;
    data.top = clouds.top;
    data.tileSize = 1.0f;
    data.gridX = data.gridY = data.gridZ = 2;
    data.cellsX = data.cellsY = data.cellsZ = 1;
    data.detailPeriod = 1.0f;
    if (!clouds.enabled)
    {
        return volume;
    }
    AIR_CHECK(clouds.top > clouds.bottom && clouds.scale > 0.0f, "clouds: empty layer or scale");

    const float scale = clouds.scale;
    const float thickness = clouds.top - clouds.bottom;
    const u32 n = kGridXZ;
    const u32 layers = std::max(1u, u32(std::lround(thickness / (float(kCellY) * kVoxelHeight * scale))));
    data.extinction = clouds.extinction;
    data.albedo = clouds.albedo;
    data.tileSize = float(n) * kVoxel * scale;
    data.gridX = n;
    data.gridY = layers * kCellY;
    data.gridZ = n;
    data.cellsX = n / kCellXZ;
    data.cellsY = layers;
    data.cellsZ = n / kCellXZ;
    data.enabled = 1;
    data.detailPeriod = kDetailPeriod * scale;
    data.detailStrength = kDetailStrength;
    data.sharpness = kSharpness;
    const CloudPhaseParams phase = FitCloudPhase(clouds.dropletDiameter);
    data.gHg = phase.gHg;
    data.gDraine = phase.gDraine;
    data.alphaDraine = phase.alphaDraine;
    data.wDraine = phase.wDraine;
    if (!clouds.exactPhase)
    {
        // Delta-M scaling (Wiscombe 1977): light scattered into the diffraction peak, the HG lobe
        // (a fraction 1 - wDraine of all scattering, within a fraction of a degree), goes on as if
        // it had not been scattered at all. The transport away from the forward direction stays the
        // same, while the rare paths that line up with the sun inside the peak, each carrying a
        // large share of the sun, no longer exist: far less noise, the same image a few degrees
        // away from the sun.
        const float forward = 1.0f - phase.wDraine;
        const float removed = forward * clouds.albedo;
        data.extinction = clouds.extinction * (1.0f - removed);
        data.albedo = clouds.albedo * (1.0f - forward) / (1.0f - removed);
        data.wDraine = 1.0f;
    }

    // The weather and its threshold.
    std::vector<float> weather(size_t(n) * n);
    ParallelFor(int(n),
                [&](int z)
                {
                    for (u32 x = 0; x < n; ++x)
                    {
                        weather[size_t(z) * n + x] =
                            Weather((float(x) + 0.5f) / float(n), (float(z) + 0.5f) / float(n), clouds.seed);
                    }
                });
    std::vector<float> sorted = weather;
    const float coverage = Saturate(clouds.coverage);
    const size_t rank = std::min(sorted.size() - 1, size_t(double(1.0f - coverage) * double(sorted.size())));
    std::nth_element(sorted.begin(), sorted.begin() + std::ptrdiff_t(rank), sorted.end());
    const float threshold = coverage > 0.0f ? sorted[rank] : 1e30f;
    const float highest = *std::max_element(weather.begin(), weather.end());

    // Base texels; slice y + 1 holds height y (the first and last slices stay empty).
    volume.base.assign(size_t(n) * n * (data.gridY + 2), 0);
    auto baseTexel = [&](u32 x, u32 y, u32 z) -> u8& { return volume.base[(size_t(y + 1) * n + z) * n + x]; };
    ParallelFor(int(data.gridY * n),
                [&](int row)
                {
                    const u32 y = u32(row) / n;
                    const u32 z = u32(row) % n;
                    const float h = (float(y) + 0.5f) / float(data.gridY);
                    const float height = h * thickness;
                    for (u32 x = 0; x < n; ++x)
                    {
                        const float excess = weather[size_t(z) * n + x] - threshold;
                        if (excess <= 0.0f)
                        {
                            continue;
                        }
                        const float strength = Saturate(excess / std::max(highest - threshold, 1e-6f));
                        const float cloudTop = std::pow(strength, 0.6f); // of the layer
                        const float r = h / cloudTop;
                        if (r >= 1.0f)
                        {
                            continue;
                        }
                        // Flat base at the condensation level, a dome above it.
                        const float envelope =
                            Smoothstep(0.0f, 0.025f, h) * (1.0f - Smoothstep(0.3f, 1.0f, r)) * std::sqrt(strength);
                        if (envelope <= 0.0f)
                        {
                            continue;
                        }
                        const float u = (float(x) + 0.5f) / float(n);
                        const float w = (float(z) + 0.5f) / float(n);
                        const float billows = Billows(u, height, w, data.tileSize, clouds.seed);
                        const float carve = (1.0f - billows) * 0.75f;
                        // Water grows with height above the base. (The lookup sharpens the
                        // outline after the detail has eroded it.)
                        float density = Saturate((envelope - carve) / (1.0f - carve));
                        density *= 0.6f + 0.4f * Smoothstep(0.0f, 0.6f, r);
                        baseTexel(x, y, z) = Texel(density);
                    }
                });

    // Detail texels: cellular noise repeating in all three axes, high between its points.
    volume.detailSize = kDetailGrid;
    volume.detail.assign(size_t(kDetailGrid) * kDetailGrid * kDetailGrid, 0);
    ParallelFor(int(kDetailGrid * kDetailGrid),
                [&](int row)
                {
                    const u32 y = u32(row) / kDetailGrid;
                    const u32 z = u32(row) % kDetailGrid;
                    for (u32 x = 0; x < kDetailGrid; ++x)
                    {
                        const float g = float(kDetailGrid);
                        const float px = (float(x) + 0.5f) / g;
                        const float py = (float(y) + 0.5f) / g;
                        const float pz = (float(z) + 0.5f) / g;
                        float detail = 0.0f;
                        const float weights[3] = {0.625f, 0.25f, 0.125f};
                        for (int octave = 0, period = 4; octave < 3; ++octave, period *= 2)
                        {
                            const float p = float(period);
                            detail += weights[octave] *
                                      Worley(px * p, py * p, pz * p, period, period, clouds.seed + 201u + u32(octave));
                        }
                        volume.detail[(size_t(y) * kDetailGrid + z) * kDetailGrid + x] = Texel(detail);
                    }
                });

    // Majorants: the largest density among the texels any point of the cell interpolates, which
    // reach one texel beyond the cell on every side.
    volume.majorants.assign(size_t(data.cellsX) * data.cellsY * data.cellsZ, 0.0f);
    u32 coveredColumns = 0;
    std::vector<u8> columnMax(size_t(n) * n, 0);
    for (u32 z = 0; z < n; ++z)
    {
        for (u32 x = 0; x < n; ++x)
        {
            u32 m = 0;
            for (u32 y = 0; y < data.gridY; ++y)
            {
                m = std::max(m, u32(baseTexel(x, y, z)));
            }
            columnMax[size_t(z) * n + x] = u8(m);
            coveredColumns += m > 0 ? 1u : 0u;
        }
    }
    volume.coverage = float(coveredColumns) / float(n * n);
    ParallelFor(int(data.cellsY * data.cellsZ),
                [&](int row)
                {
                    const u32 cy = u32(row) / data.cellsZ;
                    const u32 cz = u32(row) % data.cellsZ;
                    for (u32 cx = 0; cx < data.cellsX; ++cx)
                    {
                        u32 m = 0;
                        const int y0 = std::max(0, int(cy * kCellY) - 1);
                        const int y1 = std::min(int(data.gridY) - 1, int((cy + 1) * kCellY));
                        for (int vz = int(cz * kCellXZ) - 1; vz <= int((cz + 1) * kCellXZ); ++vz)
                        {
                            for (int vx = int(cx * kCellXZ) - 1; vx <= int((cx + 1) * kCellXZ); ++vx)
                            {
                                const size_t column = size_t(Wrap(vz, int(n))) * n + size_t(Wrap(vx, int(n)));
                                if (columnMax[column] <= m)
                                {
                                    continue;
                                }
                                for (int vy = y0; vy <= y1; ++vy)
                                {
                                    m = std::max(m, u32(volume.base[(size_t(vy + 1) * n) * n + column]));
                                }
                            }
                        }
                        // As the lookup computes it, with a margin for rounding.
                        volume.majorants[(size_t(cy) * data.cellsZ + cz) * data.cellsX + cx] =
                            data.extinction * std::min(float(m) * (1.0f / 255.0f) * data.sharpness, 1.0f) * 1.00001f;
                    }
                });
    return volume;
}

} // namespace air::renderer
