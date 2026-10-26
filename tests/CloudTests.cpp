// Volumetric clouds (renderer::BuildCloudVolume, shader/CloudLookup.hlsli): the phase function is
// normalized and sampled exactly, the majorants bound the density everywhere a ray looks, and delta
// and ratio tracking estimate the transmittance a fine ray march computes.

#include "TestFramework.h"

#include "air/renderer/Clouds.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

using namespace air;
using namespace air::hlsl;
using namespace air::gpu;

namespace
{

// CloudLookup.hlsli over a volume in memory, as the CPU reference renderer reads it.
struct CloudReader
{
    const renderer::CloudVolume& volume;

    float CloudBaseSample(CloudData c, float3 uvw) const
    {
        return renderer::SampleTexture3D(volume.base.data(), c.gridX, c.gridZ, c.gridY + 2, uvw.x, uvw.y, uvw.z);
    }
    float CloudDetailSample(CloudData /*c*/, float3 uvw) const
    {
        const u32 n = volume.detailSize;
        return renderer::SampleTexture3D(volume.detail.data(), n, n, n, uvw.x, uvw.y, uvw.z);
    }
    float CloudMajorantLoad(CloudData /*c*/, uint cell) const { return volume.majorants[cell]; }

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

CloudData PhaseOnly(float diameter)
{
    const renderer::CloudPhaseParams p = renderer::FitCloudPhase(diameter);
    CloudData c{};
    c.gHg = p.gHg;
    c.gDraine = p.gDraine;
    c.alphaDraine = p.alphaDraine;
    c.wDraine = p.wDraine;
    return c;
}

// 2 pi times the integral of the phase function over cosines from -1 to u, in double precision on
// a grid that crowds towards u = 1, where the diffraction peak is (u = 1 - 2 t^2).
double PhaseCdf(CloudReader& reader, const CloudData& c, double u)
{
    constexpr int kSteps = 200000;
    const double tEnd = std::sqrt((1.0 - u) / 2.0);
    double sum = 0.0;
    for (int i = 0; i < kSteps; ++i)
    {
        const double t = tEnd + (1.0 - tEnd) * (i + 0.5) / kSteps;
        sum += reader.CloudPhase(c, float(1.0 - 2.0 * t * t)) * 4.0 * t * (1.0 - tEnd) / kSteps;
    }
    return 2.0 * std::numbers::pi * sum;
}

renderer::CloudVolume& DefaultVolume()
{
    static renderer::CloudVolume volume = []
    {
        const auto start = std::chrono::steady_clock::now();
        renderer::CloudVolume v = renderer::BuildCloudVolume({.enabled = true});
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("  cloud volume %ux%ux%u baked in %.0f ms, coverage %.3f\n", v.data.gridX, v.data.gridY,
                    v.data.gridZ, ms, v.coverage);
        return v;
    }();
    return volume;
}

} // namespace

TEST(Clouds_PhaseFitFollowsTheDropletSize)
{
    // Jendersie and d'Eon 2023, eq. 4-7, at 20 micrometres.
    const renderer::CloudPhaseParams p = renderer::FitCloudPhase(20.0f);
    EXPECT(Near(p.gHg, 0.99458, 1e-4, "g_HG(20)"));
    EXPECT(Near(p.gDraine, 0.59381, 1e-4, "g_D(20)"));
    EXPECT(Near(p.alphaDraine, 27.112, 1e-2, "alpha(20)"));
    EXPECT(Near(p.wDraine, 0.49814, 1e-4, "w_D(20)"));
    // Outside the fitted range the size clamps.
    const renderer::CloudPhaseParams tiny = renderer::FitCloudPhase(1.0f);
    const renderer::CloudPhaseParams five = renderer::FitCloudPhase(5.0f);
    EXPECT(tiny.gHg == five.gHg && tiny.wDraine == five.wDraine);
}

TEST(Clouds_PhaseIsNormalized)
{
    renderer::CloudVolume none;
    CloudReader reader{none};
    for (float diameter : {5.0f, 20.0f, 50.0f})
    {
        const CloudData c = PhaseOnly(diameter);
        EXPECT(Near(PhaseCdf(reader, c, 1.0), 1.0, 1e-3, "phase integral over the sphere"));
        // Strongly forward: the mean cosine of cloud droplets is about 0.85.
        double mean = 0.0;
        constexpr int kSteps = 200000;
        for (int i = 0; i < kSteps; ++i)
        {
            const double t = (i + 0.5) / kSteps;
            const double u = 1.0 - 2.0 * t * t;
            mean += 2.0 * std::numbers::pi * u * reader.CloudPhase(c, float(u)) * 4.0 * t / kSteps;
        }
        EXPECT(mean > 0.75 && mean < 0.92);
    }
}

TEST(Clouds_PhaseSamplingFollowsThePhase)
{
    renderer::CloudVolume none;
    CloudReader reader{none};
    for (float diameter : {5.0f, 20.0f, 50.0f})
    {
        const CloudData c = PhaseOnly(diameter);
        const float3 axis = normalize(float3(0.3f, -0.8f, 0.52f));
        constexpr int kSamples = 400000;
        const double cuts[] = {-0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999, 0.9999};
        int below[std::size(cuts)] = {};
        uint seed = 7u;
        for (int i = 0; i < kSamples; ++i)
        {
            const float3 u = CloudReader::Rand3(seed);
            const float3 w = reader.SampleCloudPhase(c, axis, u);
            EXPECT(std::abs(length(w) - 1.0f) < 1e-4f);
            const double cosine = dot(w, axis);
            for (size_t k = 0; k < std::size(cuts); ++k)
            {
                below[k] += cosine <= cuts[k] ? 1 : 0;
            }
        }
        // The fraction of samples below each cut is the phase function's CDF there (within ~4 sigma).
        for (size_t k = 0; k < std::size(cuts); ++k)
        {
            EXPECT(Near(double(below[k]) / kSamples, PhaseCdf(reader, c, cuts[k]), 3e-3, "sampled CDF"));
        }
    }
}

TEST(Clouds_VolumeCoversTheRequestedSkyAndMajorantsBoundIt)
{
    renderer::CloudVolume& volume = DefaultVolume();
    const CloudData& c = volume.data;
    EXPECT(c.enabled == 1u);
    EXPECT(volume.base.size() == size_t(c.gridX) * c.gridZ * (c.gridY + 2));
    EXPECT(volume.detail.size() == size_t(volume.detailSize) * volume.detailSize * volume.detailSize);
    EXPECT(volume.majorants.size() == size_t(c.cellsX) * c.cellsY * c.cellsZ);
    EXPECT(volume.coverage > 0.2f && volume.coverage < 0.4f); // 0.35 asked; erosion trims the edges
    CloudReader reader{volume};

    // Along random rays, the density never exceeds the majorant of the segment the DDA gives.
    uint seed = 3u;
    int checked = 0;
    for (int ray = 0; ray < 2000; ++ray)
    {
        const float3 o =
            CloudReader::Rand3(seed) * float3(40000.0f, 4000.0f, 40000.0f) - float3(20000.0f, 0.0f, 20000.0f);
        const float z = CloudReader::Rand(seed) * 2.0f - 1.0f;
        const float phi = CloudReader::Rand(seed) * 6.2831853f;
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float3 d(r * std::cos(phi), z, r * std::sin(phi));
        CloudReader::CloudMarch m;
        if (!reader.BeginCloudMarch(c, o, d, 1e30f, m))
        {
            continue;
        }
        float t0 = 0.0f;
        float t1 = 0.0f;
        float majorant = 0.0f;
        for (int segment = 0; segment < 64 && reader.NextCloudSegment(c, m, t0, t1, majorant); ++segment)
        {
            for (int k = 0; k < 8; ++k)
            {
                const float t = t0 + (t1 - t0) * (k == 0 ? 0.0f : (k == 7 ? 1.0f : CloudReader::Rand(seed)));
                const float sigma = c.extinction * reader.CloudDensity(c, o + d * t);
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
    EXPECT(checked > 10000);

    // The density repeats with the tile horizontally at the same altitude (the layer curves with the
    // planet; up to the filter's 8-bit weights: rounding the shifted coordinate may move a weight by
    // 1/256).
    for (int i = 0; i < 200; ++i)
    {
        const float3 u = CloudReader::Rand3(seed);
        const float3 p(u.x * c.tileSize, c.bottom + u.y * (c.top - c.bottom), u.z * c.tileSize);
        float3 q = p + float3(c.tileSize, 0.0f, -2.0f * c.tileSize);
        q.y += reader.CloudAltitude(p) - reader.CloudAltitude(q);
        const float a = reader.CloudDensity(c, p);
        const float b = reader.CloudDensity(c, q);
        EXPECT(std::abs(a - b) < 1.2e-2f);
    }
    // 100 km away the layer is 785 m lower.
    EXPECT(std::abs(reader.CloudAltitude(float3(100000.0f, 0.0f, 0.0f)) - 784.8f) < 1.0f);
}

// The wind carries the density with it, and the majorant cells (anchored to the base) along.
TEST(Clouds_WindCarriesTheDensityAndItsMajorants)
{
    renderer::CloudVolume& volume = DefaultVolume();
    CloudReader reader{volume};
    const CloudData still = volume.data;
    CloudData windy = still;
    windy.baseOffsetX = 1234.5f;
    windy.baseOffsetZ = -777.25f;
    windy.detailOffsetX = windy.baseOffsetX;
    windy.detailOffsetZ = windy.baseOffsetZ;
    uint seed = 5u;
    for (int i = 0; i < 200; ++i)
    {
        const float3 u = CloudReader::Rand3(seed);
        const float3 p(u.x * still.tileSize, still.bottom + u.y * (still.top - still.bottom), u.z * still.tileSize);
        float3 q = p + float3(windy.baseOffsetX, 0.0f, windy.baseOffsetZ);
        q.y += reader.CloudAltitude(p) - reader.CloudAltitude(q);
        EXPECT(std::abs(reader.CloudDensity(still, p) - reader.CloudDensity(windy, q)) < 1.2e-2f);
    }
    // The detail drifting on over the base still only erodes it: the majorants bound it.
    windy.detailOffsetX += 333.0f;
    windy.detailOffsetY = 41.0f;
    int checked = 0;
    for (int ray = 0; ray < 500; ++ray)
    {
        const float3 o =
            CloudReader::Rand3(seed) * float3(40000.0f, 4000.0f, 40000.0f) - float3(20000.0f, 0.0f, 20000.0f);
        const float3 d = normalize(CloudReader::Rand3(seed) * 2.0f - 1.0f);
        CloudReader::CloudMarch m;
        if (!reader.BeginCloudMarch(windy, o, d, 1e30f, m))
        {
            continue;
        }
        float t0 = 0.0f;
        float t1 = 0.0f;
        float majorant = 0.0f;
        for (int segment = 0; segment < 64 && reader.NextCloudSegment(windy, m, t0, t1, majorant); ++segment)
        {
            for (int k = 0; k < 4; ++k)
            {
                const float t = t0 + (t1 - t0) * CloudReader::Rand(seed);
                EXPECT(windy.extinction * reader.CloudDensity(windy, o + d * t) <= majorant);
                ++checked;
            }
        }
    }
    EXPECT(checked > 2000);
}

// Ratio tracking's mean and delta tracking's chance to pass are both exp(-optical depth), which a
// fine ray march gives.
TEST(Clouds_TrackingMatchesARayMarch)
{
    renderer::CloudVolume& volume = DefaultVolume();
    const CloudData& c = volume.data;
    CloudReader reader{volume};
    uint seed = 11u;
    int tested = 0;
    for (int attempt = 0; attempt < 400 && tested < 6; ++attempt)
    {
        const float ox = CloudReader::Rand(seed) * c.tileSize;
        const float3 o(ox, 0.0f, CloudReader::Rand(seed) * c.tileSize);
        const float elevation = 0.3f + CloudReader::Rand(seed) * 1.2f;
        const float azimuth = CloudReader::Rand(seed) * 6.2831853f;
        const float3 d(std::cos(elevation) * std::cos(azimuth), std::sin(elevation),
                       std::cos(elevation) * std::sin(azimuth));
        // Through the whole layer, which curves with the planet (its altitude, not y).
        constexpr double kStep = 0.5;
        double depth = 0.0;
        for (double t = 0.5 * kStep; reader.CloudAltitude(o + d * float(t)) <= c.top + 1.0f; t += kStep)
        {
            depth += c.extinction * reader.CloudDensity(c, o + d * float(t)) * kStep;
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
            ratio += reader.CloudTransmittance(c, o, d, 1e30f, seed);
            float tHit = 0.0f;
            passed += reader.CloudCollision(c, o, d, 1e30f, seed, tHit) ? 0 : 1;
        }
        const double sigma = std::sqrt(expected * (1.0 - expected) / kSamples);
        EXPECT(Near(ratio / kSamples, expected, 5.0 * sigma + 2e-3, "ratio tracking"));
        EXPECT(Near(double(passed) / kSamples, expected, 5.0 * sigma + 2e-3, "delta tracking"));
    }
    EXPECT(tested == 6);
}

TEST(Clouds_DisabledCloudsAreTransparent)
{
    const renderer::CloudVolume volume = renderer::BuildCloudVolume({});
    EXPECT(volume.data.enabled == 0u && volume.base.empty() && volume.majorants.empty());
    CloudReader reader{volume};
    uint seed = 1u;
    float tHit = 0.0f;
    EXPECT(reader.CloudTransmittance(volume.data, float3(0.0f), float3(0.0f, 1.0f, 0.0f), 1e30f, seed) == 1.0f);
    EXPECT(!reader.CloudCollision(volume.data, float3(0.0f), float3(0.0f, 1.0f, 0.0f), 1e30f, seed, tHit));
}
