#pragma once

// Volumetric clouds (docs/architecture.md §8.10): a layer of cumulus over the scene, made of water
// droplets that scatter light with Mie's phase function (approximated after Jendersie and d'Eon
// 2023) and hardly absorb it. Their density is procedural (coverage from a weather map, a cumulus
// profile with height, billowy noise), baked on the CPU into 3D textures that repeat horizontally,
// read by the GPU path tracer (hardware trilinear filtering) and the CPU reference (the same filter,
// emulated) alike through shader/CloudLookup.hlsli, with a coarse grid of majorants for unbiased
// delta and ratio tracking.

#include "air/core/Types.h"
#include "air/renderer/TextureSampling.h"
#include "air/renderer/shader/CloudShared.h"

#include <vector>

namespace air::renderer
{

struct Clouds
{
    bool enabled = false;
    float bottom = 1500.0f; // cloud base, metres above the scene's y = 0
    float top = 3500.0f;    // the highest tops
    float coverage = 0.35f; // fraction of the sky the clouds cover (seen from straight below)
    // Extinction where the density is 1, per metre (cumulus: 0.02 to 0.1, a mean free path of 10 to
    // 50 m; sunlight barely gets through a kilometre of it).
    float extinction = 0.05f;
    float dropletDiameter = 20.0f; // effective, micrometres (5 to 50): shapes the phase function
    float albedo = 1.0f;           // single scattering; water absorbs almost nothing in the visible
    // false: the phase function's diffraction peak (a lobe a fraction of a degree wide holding about
    // half of all scattering) counts as unscattered light (delta-M scaling of extinction and
    // albedo), which removes most of the noise of sunlit clouds and only changes the image within
    // a degree or so of the sun. true: keep it as scattering (the reference).
    bool exactPhase = false;
    float scale = 1.0f; // size of the cloud shapes, relative
    u32 seed = 1;
    // The wind at the clouds: they drift with it in real time (the renderer's clock; still when
    // accumulating) and lean with its shear. Metres per second, and the direction it blows towards in
    // degrees from +x towards +z.
    float windSpeed = 12.0f;
    float windHeading = 0.0f;

    bool operator==(const Clouds&) const = default;
};

// The phase function's parameters for droplets of `diameter` micrometres (clamped to 5..50): the
// fit of Jendersie and d'Eon 2023, eq. 4-7.
struct CloudPhaseParams
{
    float gHg = 0.0f;
    float gDraine = 0.0f;
    float alphaDraine = 0.0f;
    float wDraine = 0.0f;
};
CloudPhaseParams FitCloudPhase(float diameter);

struct CloudVolume
{
    gpu::CloudData data{}; // enabled == 0: no clouds (and no textures); no bindless indices
    // R8_UNORM texels as CloudShared.h lays them out: base gridX x gridZ x (gridY + 2), detail
    // detailSize^3, each x fastest.
    std::vector<u8> base;
    std::vector<u8> detail;
    u32 detailSize = 0;
    std::vector<float> majorants; // cellsX * cellsZ * cellsY
    float coverage = 0.0f;        // fraction of base texel columns with any cloud
};

// Bakes the density (about 0.2 s of CPU in release builds, all threads).
CloudVolume BuildCloudVolume(const Clouds& clouds);

} // namespace air::renderer
