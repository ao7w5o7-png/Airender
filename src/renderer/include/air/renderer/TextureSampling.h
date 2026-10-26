#pragma once

// The GPU's trilinear texture filtering, emulated for the CPU renderers (the reference renderer,
// tests) that read the same tables as the GPU: texel centres at (i + 0.5) / size, interpolation
// weights with 8 fractional bits (the filtering precision D3D requires and GPUs implement).

#include "air/core/Types.h"
#include "air/renderer/shader/HlslCompat.h"

namespace air::renderer
{

// An 8-bit (R8_UNORM) 3D texture, x fastest, with wrapping addressing, at texture coordinates
// (u, v, w).
float SampleTexture3D(const u8* texels, u32 width, u32 height, u32 depth, float u, float v, float w);

// A half-float RGBA (RGBA16_FLOAT) 3D texture, 4 halves per texel, x fastest, with clamping
// addressing.
hlsl::float4 SampleTexture3DClamped(const u16* texels, u32 width, u32 height, u32 depth, float u, float v, float w);

// A half-float single-channel (R16_FLOAT) 3D texture, x fastest, with clamping addressing.
float SampleTexture3DClampedR16(const u16* texels, u32 width, u32 height, u32 depth, float u, float v, float w);

// An 8-bit (R8_UNORM) 3D texture, x fastest, with clamping addressing.
float SampleTexture3DClampedR8(const u8* texels, u32 width, u32 height, u32 depth, float u, float v, float w);

// An 8-bit RGBA (RGBA8_UNORM, or RGBA8_SRGB when srgb: the colour channels decoded before filtering)
// 2D texture, one u32 per texel (r in the low byte), x fastest, with wrapping addressing, bilinear at
// level 0.
hlsl::float4 SampleTexture2DWrapRGBA8(const u32* texels, u32 width, u32 height, bool srgb, float u, float v);

// A float RGBA (RGBA32_FLOAT) 3D texture, 4 floats per texel, x fastest, with clamping addressing.
hlsl::float4 SampleTexture3DClampedRGBA32(const float* texels, u32 width, u32 height, u32 depth, float u, float v,
                                          float w);

} // namespace air::renderer
