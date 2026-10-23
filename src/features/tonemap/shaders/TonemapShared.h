// Shared between C++ and HLSL (see air/renderer/shader/HlslCompat.h).
#ifndef AIR_TONEMAP_SHARED_H
#define AIR_TONEMAP_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::tonemap::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kOperatorClamp = 0;   // no curve, clip at 1
AIR_CONST uint kOperatorAces = 1;    // ACES RRT + sRGB ODT fit (Hill)
AIR_CONST uint kOperatorNeutral = 2; // Khronos PBR Neutral: keeps base colors faithful up to ~0.8
AIR_CONST uint kOperatorCount = 3;

AIR_CONST uint kFlagDisplayInput = 1u;     // input is already display-referred (debug views): no curve
AIR_CONST uint kFlagHighlightInvalid = 2u; // NaN / Inf -> magenta, negative -> cyan
AIR_CONST uint kFlagEncodeSrgb = 4u;       // apply the sRGB transfer function

struct TonemapConstants
{
    uint image; // RWTexture2D<float4>, tonemapped in place
    uint width;
    uint height;
    uint op;
    uint flags;
    float whitePoint; // input value mapped to display white by kOperatorClamp
    uint pad0;
    uint pad1;
};

#ifdef __cplusplus
static_assert(sizeof(TonemapConstants) == 32);
} // namespace air::tonemap::gpu
#endif

#endif
