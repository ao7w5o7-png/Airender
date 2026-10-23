// Shared between C++ and HLSL. Template for new features: copy this directory and rename.
#ifndef AIR_EXAMPLE_SHARED_H
#define AIR_EXAMPLE_SHARED_H

#ifdef __cplusplus
#include "air/core/Types.h"
namespace air::example {
using uint = air::u32;
#define AIR_EX_ALIGN alignas(16)
#else
#define AIR_EX_ALIGN
#endif

struct AIR_EX_ALIGN GenerateConstants {
    uint outputUav; // RWTexture2D<float4>
    uint width;
    uint height;
    float time;
    float scale;
    uint pad0, pad1, pad2;
};

struct AIR_EX_ALIGN CompositeConstants {
    uint patternSrv; // Texture2D<float4>
    uint outputUav;  // RWTexture2D<float4>
    uint width;
    uint height;
    float opacity;
    uint pad0, pad1, pad2;
};

#ifdef __cplusplus
static_assert(sizeof(GenerateConstants) == 32);
static_assert(sizeof(CompositeConstants) == 32);
} // namespace air::example
#endif

#endif
