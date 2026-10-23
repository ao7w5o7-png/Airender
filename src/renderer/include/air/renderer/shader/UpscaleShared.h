// FrameRunner's fallback upscale (Upscale.hlsl), shared between C++ and HLSL (see HlslCompat.h).
#ifndef AIR_RENDERER_UPSCALE_SHARED_H
#define AIR_RENDERER_UPSCALE_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

struct UpscaleConstants
{
    uint source;      // Texture2D<float4> at the render resolution
    uint destination; // RWTexture2D<float4> at the output resolution
    uint sourceWidth;
    uint sourceHeight;
    uint destinationWidth;
    uint destinationHeight;
    uint pad0;
    uint pad1;
};

#ifdef __cplusplus
static_assert(sizeof(UpscaleConstants) == 32);
} // namespace air::gpu
#endif

#endif
