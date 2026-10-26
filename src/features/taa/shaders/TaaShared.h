// Shared between TaaFeature.cpp and Taa.hlsl.
#ifndef AIR_TAA_SHARED_H
#define AIR_TAA_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::taa::gpu
{
using namespace air::hlsl;
#endif

struct TaaConstants
{
    uint color;   // Texture2D<float4>: the frame at the render resolution (sceneColor)
    uint motion;  // Texture2D<float2>: previous - current position in render pixels
    uint history; // Texture2D<float4>: last frame's result at the output resolution; kInvalidIndex: none
    uint result;  // RWTexture2D<float4>: this frame's
    uint renderWidth;
    uint renderHeight;
    uint width; // output resolution
    uint height;
    float jitterX; // the frame's samples lie at pixel + 0.5 + jitter (render pixels)
    float jitterY;
    float blend; // weight of this frame
    float clip;  // the history is clipped to the neighbourhood's mean +- clip standard deviations
};

#ifdef __cplusplus
static_assert(sizeof(TaaConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::taa::gpu
#endif

#endif
