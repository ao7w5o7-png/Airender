// The procedural cumulus behind BuildCloudVolume (docs/architecture.md §8.10).
//
// A 2D weather field decides where clouds are: gradient noise from 12.8 down to 1.6 km gathers them
// into fields with clear gaps between, and cellular noise at 3.2, 1.6 and 0.8 km, its coordinates
// warped by more gradient noise, makes clouds of many sizes and uneven outlines. Its quantile at
// 1 - coverage is the threshold, so the requested fraction of the tile lies under cloud bases; how far
// the weather exceeds it sets how tall the cloud grows and how dense it is. As in Horizon's Nubis
// (Schneider 2015, 2017) a second field, of cloud types, varies regionally: stratocumulus (shallow
// sheets, lumpy flat tops), cumulus (domes) and congestus (towers whose tops spread like anvils), each
// with its height profile. Vertically each cloud has a flat base at the condensation level and narrows
// with height, where billowy cellular noise (cells of 640 m down to 160 m), its coordinates swirled by
// curl noise, carves it, less so near the base; the wind's shear leans the clouds with height. A second periodic grid
// of finer cellular noise (1.024 km, 25 to the tile) erodes the fringe gently at render time (CloudLookup.hlsli), and
// the eroded density is sharpened into crisp outlines over even interiors. The tile (25.6 km) repeats too far away to
// show from the ground. Everything is deterministic in the seed.

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

constexpr u32 kGridXZ = 512;          // base voxels across the tile
constexpr float kVoxel = 50.0f;       // their width at scale 1, metres: a 25.6 km tile
constexpr float kVoxelHeight = 25.0f; // their height at scale 1, about
constexpr u32 kCellXZ = 8;            // base voxels per majorant cell
constexpr u32 kCellY = 4;
constexpr u32 kDetailGrid = 128;         // detail voxels along each axis
constexpr float kDetailPeriod = 1024.0f; // metres at scale 1 (the tile is 25 of them): detail voxels of 8 m
// How deep the detail erodes the fringe, and the sharpening after it: stronger made every cloud a
// cauliflower of same-sized bumps.
constexpr float kDetailStrength = 0.4f;
constexpr float kSharpness = 5.0f;

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
    // Regional: fields of clouds and clear gaps, 12.8 to 1.6 km.
    float regional = 0.0f;
    float amplitude = 0.5f;
    for (int octave = 0, period = 2; octave < 4; ++octave, period *= 2)
    {
        regional += amplitude * Perlin(u * float(period), w * float(period), period, seed + u32(octave));
        amplitude *= 0.5f;
    }
    // Cells of 3.2, 1.6 and 0.8 km over warped coordinates (about 700 m): clouds of many sizes, their
    // outlines uneven rather than round.
    const float warpU =
        0.027f * (Perlin(u * 4.0f, w * 4.0f, 4, seed + 31u) + 0.5f * Perlin(u * 16.0f, w * 16.0f, 16, seed + 32u));
    const float warpW =
        0.027f * (Perlin(u * 4.0f, w * 4.0f, 4, seed + 33u) + 0.5f * Perlin(u * 16.0f, w * 16.0f, 16, seed + 34u));
    const float x = u + warpU;
    const float z = w + warpW;
    const float cells = 0.5f * (1.0f - Worley(x * 8.0f, z * 8.0f, 8, seed + 17u)) +
                        0.3f * (1.0f - Worley(x * 16.0f, z * 16.0f, 16, seed + 18u)) +
                        0.2f * (1.0f - Worley(x * 32.0f, z * 32.0f, 32, seed + 19u));
    return 0.8f * regional + cells;
}

// The cloud type over the tile: 0 stratocumulus, 0.5 cumulus, 1 congestus; fields of 12.8 to 3.2 km.
float CloudType(float u, float w, u32 seed)
{
    const float t =
        0.5f + 1.4f * Perlin(u * 2.0f, w * 2.0f, 2, seed + 51u) + 0.5f * Perlin(u * 8.0f, w * 8.0f, 8, seed + 52u);
    return Saturate(t);
}

// A divergence-free swirl over the tile (the curl of gradient noise of 1.6 km cells): an offset in
// tile units, about +-1 at most.
void Curl(float u, float w, u32 seed, float& du, float& dw)
{
    const float e = 1.0f / 1024.0f;
    const float px0 = Perlin((u - e) * 16.0f, w * 16.0f, 16, seed + 61u);
    const float px1 = Perlin((u + e) * 16.0f, w * 16.0f, 16, seed + 61u);
    const float pz0 = Perlin(u * 16.0f, (w - e) * 16.0f, 16, seed + 61u);
    const float pz1 = Perlin(u * 16.0f, (w + e) * 16.0f, 16, seed + 61u);
    const float scale = 1.0f / (2.0f * e * 16.0f) * 0.5f;
    du = (pz1 - pz0) * scale;
    dw = -(px1 - px0) * scale;
}

// Billows over the tile at (u, height in metres, w): cellular noise in cells of 640, 320 and 160 m,
// high near its points.
float Billows(float u, float y, float w, float tileSize, u32 seed)
{
    float sum = 0.0f;
    const float weights[3] = {0.55f, 0.3f, 0.15f};
    for (int octave = 0, period = 40; octave < 3; ++octave, period *= 2)
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

// The threshold of the cirrus' noise (CirrusOpticalDepth in CloudLookup.hlsli) that the given fraction
// of the sky exceeds: the noise is close to normal, mean 0.507 and deviation 0.228 (measured over 100 km).
float CirrusThreshold(float coverage)
{
    // The normal quantile by Acklam's rational approximation (relative error below 1.2e-9).
    const double p = std::clamp(1.0 - double(coverage), 1e-6, 1.0 - 1e-6);
    constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,  -2.759285104469687e+02,
                            1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                            6.680131188771972e+01, -1.328068155288572e+01};
    constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                            -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
    double z = 0.0;
    if (p < 0.02425)
    {
        const double q = std::sqrt(-2.0 * std::log(p));
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else if (p > 1.0 - 0.02425)
    {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else
    {
        const double q = p - 0.5;
        const double r = q * q;
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    return float(0.507 + 0.228 * z);
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
    const float heading = clouds.windHeading * (3.14159265f / 180.0f);
    const float windU = std::cos(heading);
    const float windW = std::sin(heading);
    data.windX = clouds.windSpeed * windU;
    data.windZ = clouds.windSpeed * windW;
    data.cirrusAltitude = clouds.cirrusAltitude;
    data.cirrusDepth = clouds.cirrus > 0.0f ? clouds.cirrusDepth : 0.0f;
    data.cirrusThreshold = CirrusThreshold(clouds.cirrus);
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
    // The cirrus' ice scatters with the same phase function, scaled alike.
    data.cirrusDepth *= data.extinction / clouds.extinction;

    // The weather and its threshold.
    std::vector<float> weather(size_t(n) * n);
    ParallelFor(int(n),
                [&](int z)
                {
                    for (u32 x = 0; x < n; ++x)
                    {
                        const float u = (float(x) + 0.5f) / float(n);
                        const float w = (float(z) + 0.5f) / float(n);
                        // Stratocumulus spreads into sheets; congestus gathers into fewer, larger cells.
                        const float type = CloudType(u, w, clouds.seed);
                        const float sheets = 0.35f * Saturate(1.0f - type * 2.5f);
                        const float towers = Smoothstep(0.65f, 1.0f, type);
                        const float big = 1.0f - Worley(u * 4.0f, w * 4.0f, 4, clouds.seed + 71u);
                        weather[size_t(z) * n + x] =
                            Weather(u, w, clouds.seed) + sheets + towers * (0.9f * big - 0.45f);
                    }
                });
    std::vector<float> sorted = weather;
    const float coverage = Saturate(clouds.coverage);
    const size_t rank = std::min(sorted.size() - 1, size_t(double(1.0f - coverage) * double(sorted.size())));
    std::nth_element(sorted.begin(), sorted.begin() + std::ptrdiff_t(rank), sorted.end());
    const float threshold = coverage > 0.0f ? sorted[rank] : 1e30f;
    const float highest = *std::max_element(weather.begin(), weather.end());

    // The types, and the weather between its texels (repeating).
    std::vector<float> types(size_t(n) * n);
    ParallelFor(int(n),
                [&](int z)
                {
                    for (u32 x = 0; x < n; ++x)
                    {
                        types[size_t(z) * n + x] =
                            CloudType((float(x) + 0.5f) / float(n), (float(z) + 0.5f) / float(n), clouds.seed);
                    }
                });
    auto bilinear = [&](const std::vector<float>& field, float u, float w)
    {
        const float fx = u * float(n) - 0.5f;
        const float fz = w * float(n) - 0.5f;
        const int x0 = int(std::floor(fx));
        const int z0 = int(std::floor(fz));
        const float ax = fx - float(x0);
        const float az = fz - float(z0);
        auto at = [&](int x, int z) { return field[size_t(Wrap(z, int(n))) * n + size_t(Wrap(x, int(n)))]; };
        return (at(x0, z0) * (1.0f - ax) + at(x0 + 1, z0) * ax) * (1.0f - az) +
               (at(x0, z0 + 1) * (1.0f - ax) + at(x0 + 1, z0 + 1) * ax) * az;
    };
    // The wind's shear: 500 m of lean over the layer's height downwind, 150 m across (veering).
    const float shearU = (500.0f * windU - 150.0f * windW) / data.tileSize;
    const float shearW = (500.0f * windW + 150.0f * windU) / data.tileSize;
    // The swirl: up to about 200 m, more higher up.
    const float swirl = 200.0f / data.tileSize;

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
                        // Where this texel's air came from: leaned by the shear, swirled by the curl.
                        float u = (float(x) + 0.5f) / float(n) - shearU * h;
                        float w = (float(z) + 0.5f) / float(n) - shearW * h;
                        float du = 0.0f;
                        float dw = 0.0f;
                        Curl(u, w, clouds.seed, du, dw);
                        u += du * swirl * (0.3f + h);
                        w += dw * swirl * (0.3f + h);
                        const float excess = bilinear(weather, u, w) - threshold;
                        if (excess <= 0.0f)
                        {
                            continue;
                        }
                        const float strength = Saturate(excess / std::max(highest - threshold, 1e-6f));
                        const float type = bilinear(types, u, w);
                        // Of the layer, by type: stratocumulus stays low, cumulus mostly shallow (the
                        // strongest tall), congestus towers.
                        const float shallow = 0.06f + 0.18f * strength;
                        const float cumulus = 0.12f + 0.88f * std::pow(strength, 1.6f);
                        const float tower = 0.3f + 0.7f * std::pow(strength, 0.7f);
                        const float cloudTop = type < 0.5f ? shallow + (cumulus - shallow) * (type * 2.0f)
                                                           : cumulus + (tower - cumulus) * (type * 2.0f - 1.0f);
                        const float r = h / cloudTop;
                        if (r >= 1.0f)
                        {
                            continue;
                        }
                        // Flat base at the condensation level; above it the cloud narrows (stratocumulus
                        // hardly: flat-topped sheets) into a dome, and a tower's top spreads into an anvil.
                        const float narrowing = 0.05f + 0.3f * Smoothstep(0.0f, 0.5f, type);
                        const float anvil = 0.35f * Smoothstep(0.7f, 1.0f, type) * Smoothstep(0.65f, 0.95f, r);
                        const float envelope = Smoothstep(0.0f, 0.03f, h) * (1.0f - Smoothstep(0.45f, 1.0f, r)) *
                                               Smoothstep(0.0f, 0.15f, strength - narrowing * r * r + anvil) *
                                               std::sqrt(strength);
                        if (envelope <= 0.0f)
                        {
                            continue;
                        }
                        const float billows = Billows(u, height, w, data.tileSize, clouds.seed);
                        // Billows carve the flanks and the top; the base stays flat.
                        const float carve = (1.0f - billows) * (0.2f + 0.35f * Smoothstep(0.0f, 0.5f, r));
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
                        for (int octave = 0, period = 6; octave < 3; ++octave, period *= 2)
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
