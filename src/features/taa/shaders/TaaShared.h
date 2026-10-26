// Shared between TaaFeature.cpp and Taa.hlsli's passes (TaaFast.hlsl, Taa.hlsl).
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
    uint result;  // RWTexture2D<float4>: this frame's (rgb; a: the mean of the frame's squared luminance)
    // The fast history: a short average of the frames that only follows the camera, whose
    // neighbourhood the result's history is clipped to (TaaFast.hlsl writes fastResult, Taa.hlsl reads it).
    uint fastHistory; // Texture2D<float4>: last frame's; kInvalidIndex: none
    uint fastResult;  // RWTexture2D<float4> (TaaFast.hlsl) / Texture2D<float4> (Taa.hlsl): this frame's
    uint renderWidth;
    uint renderHeight;
    uint width; // output resolution
    uint height;
    float jitterX; // the frame's samples lie at pixel + 0.5 + jitter (render pixels)
    float jitterY;
    float blend;     // weight of this frame in the result
    float fastBlend; // ... in the fast history
    float clip;      // the history is clipped to the fast history's mean +- clip deviations
    uint frame;      // seeds the stochastic rounding to half precision
    // 1 when the camera moved since the last frame: every pixel counts as moving (what lies in front of
    // the sky, a volume's light, has the sky's motion vectors, which a moving camera does not move).
    float cameraMoved;
};

#ifdef __cplusplus
static_assert(sizeof(TaaConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::taa::gpu
#endif

#endif
