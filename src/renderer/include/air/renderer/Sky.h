#pragma once

// The physically based sky (docs/architecture.md §8.9): the sun outside an Earth-like atmosphere
// (Rayleigh and Mie scattering, ozone absorption), computed spectrally on the CPU with the
// transmittance and multiple-scattering tables of Hillaire 2020 and kept as one table of sky
// radiance over all directions as seen from the scene (shader/SkyShared.h), plus the sun's radiance
// after the atmosphere, plus the aerial perspective: what the air between the scene and anything
// up to 100 km away adds and takes away. The GPU path tracer and the CPU reference read the same
// tables.

#include "air/core/Types.h"
#include "air/renderer/shader/SkyShared.h"

#include <vector>

namespace air::renderer
{

struct Sky
{
    bool physical = false;            // false: the scene's uniform `environment`, no sun
    float sunElevation = 30.0f;       // degrees above the horizon (negative: twilight)
    float sunAzimuth = 0.0f;          // degrees, from +z towards +x
    float sunAngularRadius = 0.2665f; // degrees
    float altitude = 0.0f;            // of the scene above sea level, metres
    // Aerosol (haze) optical depth at 550 nm: 0.005 pristine, 0.05 a clear day, 0.3 and up hazy.
    float aerosolDepth = 0.05f;
    float groundAlbedo = 0.1f; // of the planet beyond the scene, seen below the horizon

    bool operator==(const Sky&) const = default;
};

// Unit vector towards the sun (y up).
hlsl::float3 SunDirection(const Sky& sky);

struct SkyTable
{
    gpu::SkyData data{};         // all zero: without a physical sky only the sun and the uniform radiance are set
    std::vector<float> radiance; // data.width * data.height texels, 3 floats each (linear sRGB, nits)
    // Aerial perspective (SkyShared.h): aerialWidth * aerialHeight * data.aerialSlices RGBA16F texels
    // each, x fastest; empty without a physical sky.
    u32 aerialWidth = 0;
    u32 aerialHeight = 0;
    std::vector<u16> aerialInscatter;
    std::vector<u16> aerialTransmittance;
};

// The table and sun of a physical sky (about 0.1 s of CPU in release builds); for any other sky a
// uniform one of radiance `uniform` and no table.
SkyTable BuildSkyTable(const Sky& sky, hlsl::float3 uniform);

// Quantities of the model behind the table, for tests.
struct SkyModelProbe
{
    float zenithOpticalDepth550 = 0.0f;  // Rayleigh + Mie + ozone from sea level to space, 550 nm
    float zenithTransmittance550 = 0.0f; // the same through the model's transmittance table
    float sunIlluminanceOutside = 0.0f;  // perpendicular, above the atmosphere, lux
};
SkyModelProbe ProbeSkyModel(const Sky& sky);

} // namespace air::renderer
