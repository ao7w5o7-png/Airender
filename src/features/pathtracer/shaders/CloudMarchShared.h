// Shared between CloudMarcher.cpp and the real-time clouds' passes (CloudShadow.hlsl, CloudSky.hlsl,
// CloudMarch.hlsl, CloudReconstruct.hlsl; docs/architecture.md §8.19).
#ifndef AIR_PT_CLOUD_MARCH_SHARED_H
#define AIR_PT_CLOUD_MARCH_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
#endif

// The layer the camera's rays see is reconstructed at the render's size; every frame the march lights
// one pixel of every 4 x 4 block of it, so that each is new every 16 frames.
AIR_CONST uint kCloudShadowMapSize = 512; // texels over the clouds' tile, each axis
AIR_CONST uint kCloudSkyMapSize = 128;    // texels over the upper hemisphere, each axis

struct CloudMarchConstants
{
    uint frameData; // StructuredBuffer<FrameData>: the camera, the sky and the clouds
    uint frameSlot;
    uint skyTable;    // StructuredBuffer<float> (SkyShared.h)
    uint target;      // RWTexture2D: R32F shadow map, RGBA32F sky map / march / layer
    uint targetDepth; // RWTexture2D<float>: the march's and the layer's front distance
    uint width;       // of the target
    uint height;
    uint layerWidth; // the reconstructed layer (the render's size)
    uint layerHeight;
    uint offset;       // this frame's pixel of each 4 x 4 block of the layer: x | y << 2
    uint marched;      // Texture2D<float4>: this frame's march (reconstruction)
    uint marchedDepth; // Texture2D<float>
    uint history;      // Texture2D<float4>: last frame's layer; kInvalidIndex: none
    uint historyDepth; // Texture2D<float>
    uint steps;        // samples a camera ray may take in the clouds
    uint lightSamples; // samples of the optical depth towards the sun
    float windStepX;   // metres the wind carried the clouds since the last frame (reconstruction)
    float windStepZ;
};

#ifdef __cplusplus
static_assert(sizeof(CloudMarchConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pt::gpu
#endif

#endif
