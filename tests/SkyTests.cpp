// The physically based sky (renderer::BuildSkyTable): the model's optical depth and solar
// illuminance match their closed forms, daylight falls in measured ranges, the colours follow the
// sun's elevation, and the lookup the renderers use returns the table.

#include "TestFramework.h"

#include "air/renderer/Scene.h"
#include "air/renderer/Sky.h"
#include "air/renderer/TextureSampling.h"
#include "air/renderer/shader/SkyShared.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>

using namespace air;
using namespace air::hlsl;
using namespace air::gpu;

namespace
{

// SkyLookup.hlsli over a table in memory, as the CPU reference renderer reads it.
struct SkyReader
{
    const renderer::SkyTable& table;

    float3 SkyTableLoad(uint texel) const
    {
        return float3(table.radiance[size_t(texel) * 3], table.radiance[size_t(texel) * 3 + 1],
                      table.radiance[size_t(texel) * 3 + 2]);
    }
    float4 AerialInscatterSample(SkyData s, float3 uvw) const
    {
        return renderer::SampleTexture3DClamped(table.aerialInscatter.data(), table.aerialWidth, table.aerialHeight,
                                                s.aerialSlices, uvw.x, uvw.y, uvw.z);
    }
    float4 AerialTransmittanceSample(SkyData s, float3 uvw) const
    {
        return renderer::SampleTexture3DClamped(table.aerialTransmittance.data(), table.aerialWidth, table.aerialHeight,
                                                s.aerialSlices, uvw.x, uvw.y, uvw.z);
    }

// NOLINTBEGIN(readability-identifier-naming)
#include "air/renderer/shader/SkyLookup.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

float Luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

float3 Direction(double elevationDegrees, double azimuthDegrees)
{
    const double e = elevationDegrees * std::numbers::pi / 180.0;
    const double a = azimuthDegrees * std::numbers::pi / 180.0;
    return float3(float(std::cos(e) * std::sin(a)), float(std::sin(e)), float(std::cos(e) * std::cos(a)));
}

// Illuminance on a horizontal surface, lux: from the sky (the table integrated over the upper
// hemisphere) and from the sun's disc.
struct Illuminance
{
    double sky = 0.0;
    double sun = 0.0;
};

Illuminance Horizontal(const renderer::SkyTable& table)
{
    SkyReader reader{table};
    constexpr int kTheta = 128;
    constexpr int kPhi = 256;
    const double dTheta = 0.5 * std::numbers::pi / kTheta;
    const double dPhi = 2.0 * std::numbers::pi / kPhi;
    Illuminance e;
    for (int i = 0; i < kTheta; ++i)
    {
        const double theta = (i + 0.5) * dTheta; // from the zenith
        for (int j = 0; j < kPhi; ++j)
        {
            const double phi = (j + 0.5) * dPhi;
            const float3 d(float(std::sin(theta) * std::sin(phi)), float(std::cos(theta)),
                           float(std::sin(theta) * std::cos(phi)));
            e.sky += Luminance(reader.SkyRadiance(table.data, d)) * std::cos(theta) * std::sin(theta) * dTheta * dPhi;
        }
    }
    e.sun = Luminance(table.data.sunRadiance) * reader.SunSolidAngle(table.data) *
            std::max(table.data.sunDirection.y, 0.0f);
    return e;
}

bool InRange(double value, double lo, double hi, const char* what)
{
    if (value >= lo && value <= hi)
    {
        return true;
    }
    std::fprintf(stderr, "  %s = %.4g, expected %.4g .. %.4g\n", what, value, lo, hi);
    return false;
}

} // namespace

TEST(Sky_ModelMatchesClosedForms)
{
    const renderer::SkyModelProbe probe = renderer::ProbeSkyModel({.physical = true});
    // Rayleigh about 0.11, ozone about 0.03, aerosols 0.05 (the default) straight up at 550 nm.
    EXPECT(InRange(probe.zenithOpticalDepth550, 0.17, 0.21, "zenith optical depth at 550 nm"));
    // The transmittance table holds exp(-depth).
    EXPECT(InRange(probe.zenithTransmittance550, std::exp(-probe.zenithOpticalDepth550) - 0.003,
                   std::exp(-probe.zenithOpticalDepth550) + 0.003, "zenith transmittance at 550 nm"));
    // The measured solar spectrum carries the solar constant's illuminance, about 128 klux.
    EXPECT(InRange(probe.sunIlluminanceOutside, 120000.0, 136000.0, "solar illuminance outside the atmosphere"));
}

TEST(Sky_DaylightFallsInMeasuredRanges)
{
    const auto start = std::chrono::steady_clock::now();
    const renderer::SkyTable noon = renderer::BuildSkyTable({.physical = true, .sunElevation = 60.0f}, float3(0.0f));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("  sky table %ux%u built in %.0f ms\n", noon.data.width, noon.data.height, ms);
    EXPECT(noon.data.width > 0 && noon.radiance.size() == size_t(noon.data.width) * noon.data.height * 3);
    // The sun outside the atmosphere is about 1.9e9 nits; at 60 degrees about 80% gets through.
    EXPECT(InRange(Luminance(noon.data.sunRadiance), 1.3e9, 1.75e9, "sun luminance at 60 degrees"));
    // A clear day with the sun 60 degrees high: ~85-95 klux direct and 10-25 klux from the sky on
    // a horizontal surface.
    const Illuminance e = Horizontal(noon);
    EXPECT(InRange(e.sun, 75000.0, 100000.0, "horizontal sun illuminance"));
    EXPECT(InRange(e.sky, 8000.0, 30000.0, "horizontal sky illuminance"));
    SkyReader reader{noon};
    const float3 zenith = reader.SkyRadiance(noon.data, float3(0.0f, 1.0f, 0.0f));
    EXPECT(InRange(Luminance(zenith), 1500.0, 12000.0, "zenith luminance"));
    EXPECT(zenith.z > 1.3f * zenith.x); // blue
    // Brighter towards the sun (forward scattering by aerosols) than away from it.
    EXPECT(Luminance(reader.SkyRadiance(noon.data, Direction(50.0, 0.0))) >
           Luminance(reader.SkyRadiance(noon.data, Direction(50.0, 180.0))));
}

TEST(Sky_LowSunReddensAndTwilightHasNoSun)
{
    const renderer::SkyTable noon = renderer::BuildSkyTable({.physical = true, .sunElevation = 60.0f}, float3(0.0f));
    const renderer::SkyTable low = renderer::BuildSkyTable({.physical = true, .sunElevation = 2.0f}, float3(0.0f));
    const float3 sun = low.data.sunRadiance;
    EXPECT(sun.x > sun.y && sun.y > sun.z); // the path through the air removes the blue first
    EXPECT(Luminance(sun) < 0.5f * Luminance(noon.data.sunRadiance));
    // The horizon below the sun turns warmer than the zenith.
    SkyReader reader{low};
    const float3 horizon = reader.SkyRadiance(low.data, Direction(3.0, 0.0));
    const float3 zenith = reader.SkyRadiance(low.data, float3(0.0f, 1.0f, 0.0f));
    EXPECT(horizon.x / horizon.z > 2.0f * zenith.x / zenith.z);

    const renderer::SkyTable twilight =
        renderer::BuildSkyTable({.physical = true, .sunElevation = -4.0f}, float3(0.0f));
    EXPECT(Luminance(twilight.data.sunRadiance) == 0.0f);
    SkyReader dusk{twilight};
    EXPECT(InRange(Luminance(dusk.SkyRadiance(twilight.data, float3(0.0f, 1.0f, 0.0f))), 0.1, 300.0,
                   "zenith luminance in civil twilight"));
}

TEST(Sky_LookupReturnsTheTable)
{
    const renderer::SkyTable table =
        renderer::BuildSkyTable({.physical = true, .sunElevation = 25.0f, .sunAzimuth = 70.0f}, float3(0.0f));
    SkyReader reader{table};
    const u32 width = table.data.width;
    const u32 height = table.data.height;
    // At texel centres the bilinear lookup returns the texel.
    for (u32 y : {3u, height / 3, height / 2 + 7, height - 4})
    {
        for (u32 x : {2u, width / 2, width - 3})
        {
            const double u = (x + 0.5) / width;
            const double v = (y + 0.5) / height;
            const double w = 2.0 * v - 1.0;
            const double elevation = (w < 0.0 ? -1.0 : 1.0) * 90.0 * w * w;
            const double phi = 180.0 * u * u;
            const float3 looked = reader.SkyRadiance(table.data, Direction(elevation, 70.0 + phi));
            const size_t texel = (size_t(y) * width + x) * 3;
            const float3 stored(table.radiance[texel], table.radiance[texel + 1], table.radiance[texel + 2]);
            EXPECT(std::abs(Luminance(looked) - Luminance(stored)) <= 2e-3f * Luminance(stored) + 1e-3f);
        }
    }
    // Mirror symmetry about the sun's vertical plane.
    const float a = Luminance(reader.SkyRadiance(table.data, Direction(20.0, 70.0 + 40.0)));
    const float b = Luminance(reader.SkyRadiance(table.data, Direction(20.0, 70.0 - 40.0)));
    EXPECT(std::abs(a - b) <= 1e-4f * a);
}

TEST(Sky_UniformSkyHasNoTableOrSun)
{
    const renderer::SkyTable table = renderer::BuildSkyTable({}, float3(1.0f, 2.0f, 3.0f));
    EXPECT(table.data.width == 0 && table.radiance.empty());
    SkyReader reader{table};
    const float3 sky = reader.SkyRadiance(table.data, float3(0.0f, 1.0f, 0.0f));
    EXPECT(sky.x == 1.0f && sky.y == 2.0f && sky.z == 3.0f);
    EXPECT(!reader.HasSun(table.data));
}

// The sun's disc is sampled uniformly: every sample falls inside it, and their mean is its axis.
TEST(Sky_SunSamplesStayInTheDisc)
{
    renderer::SkyTable table = renderer::BuildSkyTable({}, float3(0.0f));
    table.data.sunDirection = normalize(float3(0.3f, 0.8f, -0.5f));
    table.data.sunCosRadius = std::cos(3.0f * std::numbers::pi_v<float> / 180.0f);
    SkyReader reader{table};
    float3 mean(0.0f);
    constexpr int kSide = 32;
    for (int i = 0; i < kSide; ++i)
    {
        for (int j = 0; j < kSide; ++j)
        {
            const float3 d = reader.SampleSunDirection(table.data, float2((i + 0.5f) / kSide, (j + 0.5f) / kSide));
            EXPECT(std::abs(length(d) - 1.0f) < 1e-4f);
            EXPECT(dot(d, table.data.sunDirection) >= table.data.sunCosRadius - 1e-6f);
            mean = mean + d;
        }
    }
    EXPECT(dot(normalize(mean), table.data.sunDirection) > 0.99999f);
}

// Aerial perspective: nothing at the viewer; looking straight up, the air's light over the whole
// atmosphere is the sky's; along the ground the air dims light as its extinction says, blue more
// than red, and adds more light the farther the view reaches.
TEST(Sky_AerialPerspectiveMatchesTheSkyAndTheAir)
{
    const renderer::SkyTable table =
        renderer::BuildSkyTable({.physical = true, .sunElevation = 35.0f, .sunAzimuth = 20.0f}, float3(0.0f));
    SkyReader reader{table};
    EXPECT(table.data.aerialDistance > 0.0f && table.aerialInscatter.size() == size_t(table.aerialWidth) *
                                                                                   table.aerialHeight *
                                                                                   table.data.aerialSlices * 4);
    const float3 horizontal = Direction(0.5, 110.0);
    const auto start = reader.AerialPerspective(table.data, horizontal, 0.0f);
    EXPECT(Luminance(start.inscatter) == 0.0f && start.transmittance.y == 1.0f);

    // Looking up through the whole atmosphere, the air's light is the sky's: within 1% away from the
    // sun and the zenith (there the coarser table's texels are wide: 2% and 3%).
    for (double elevation : {75.0, 60.0, 45.0, 30.0})
    {
        for (double azimuth : {80.0, 140.0, 200.0})
        {
            const float3 d = Direction(elevation, azimuth);
            const float sky = Luminance(reader.SkyRadiance(table.data, d));
            const float air = Luminance(reader.AerialPerspective(table.data, d, 100000.0f).inscatter);
            EXPECT(InRange(air, 0.99 * sky, 1.01 * sky, "air light over the whole atmosphere"));
        }
    }

    // Rayleigh scattering around 550 nm plus the aerosols (optical depth 0.05 over 1.2 km).
    const double extinction = 1.24062e-6 / std::pow(0.55, 4.0) + 0.05 / 1200.0;
    const auto far = reader.AerialPerspective(table.data, horizontal, 10000.0f);
    EXPECT(InRange(far.transmittance.y, std::exp(-extinction * 10000.0) - 0.03, std::exp(-extinction * 10000.0) + 0.03,
                   "green transmittance over 10 km along the ground"));
    EXPECT(far.transmittance.x > far.transmittance.y && far.transmittance.y > far.transmittance.z);
    float previous = 0.0f;
    for (float distance : {100.0f, 1000.0f, 5000.0f, 20000.0f, 80000.0f})
    {
        const float light = Luminance(reader.AerialPerspective(table.data, horizontal, distance).inscatter);
        EXPECT(light > previous);
        previous = light;
    }
}
