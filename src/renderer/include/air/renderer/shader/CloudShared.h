// Clouds as the renderers see them, shared by C++ and HLSL (see HlslCompat.h): a horizontal layer of
// water droplets whose density repeats horizontally (renderer::BuildCloudVolume writes it,
// docs/architecture.md §8.10). CloudLookup.hlsli reads it.
//
// The density is two 8-bit 3D textures sampled with trilinear filtering and wrapping: the base
// (gridX x gridZ x gridY texels over the tile and the layer, plus an empty slice below and above so
// that wrapping in y reads nothing) and a small detail texture repeating every detailPeriod metres
// along each axis. Texture x is world x, texture y is world z, texture z is height. The majorants
// are one float per cell, x fastest, then z, then height.
#ifndef AIR_RENDERER_CLOUD_SHARED_H
#define AIR_RENDERER_CLOUD_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

struct CloudData
{
    float bottom;     // y of the cloud base, metres
    float top;        // y of the highest tops
    float extinction; // per metre at density 1
    float albedo;     // single-scattering albedo of the droplets
    float tileSize;   // horizontal period of the base density, metres
    uint gridX;       // base density texels over the tile and the layer
    uint gridY;
    uint gridZ;
    uint cellsX; // majorant cells over the same; each covers whole texels
    uint cellsY;
    uint cellsZ;
    uint enabled;
    float detailPeriod;   // metres
    float detailStrength; // how deep the detail erodes the base density's fringe (0..1)
    uint baseTexture;     // bindless Texture3D<float> (GPU only)
    uint detailTexture;   // bindless Texture3D<float> (GPU only)
    // Phase function: Henyey-Greenstein blended with Draine's (Jendersie and d'Eon 2023).
    float gHg;
    float gDraine;
    float alphaDraine;
    float wDraine;
    uint majorants;  // bindless StructuredBuffer<float> (GPU only)
    float sharpness; // the eroded density times this, clamped to 1: crisp outlines, even interiors
    // The real-time clouds' maps (GPU only; pathtracer's CloudMaps.hlsli): the optical depth along the
    // sun from the base up (Texture2D<float> over the tile, wrapping), and what the clouds add to and
    // leave of the sky seen from the camera (Texture2D<float4>, the upper hemisphere).
    uint shadowMap;
    uint skyMap;
};

#ifdef __cplusplus
static_assert(sizeof(CloudData) == 96);
} // namespace air::gpu
#endif

#endif
