#pragma once

// Synchronous CPU <-> GPU copies for one-off data: static tables, screenshots, the CLI's --output.
// Each call submits its own command list and waits, so keep them out of the frame loop.

#include "air/core/Image.h"
#include "air/rhi/RHIDevice.h"

#include <optional>
#include <vector>

namespace air::renderer
{

// Copies mip 0 of `texture` (RGBA8/RGBA16F/RGBA32F) into a float image and waits for the GPU.
// `current` is the texture's access when called; it is left in the same state.
std::optional<Image> ReadbackTexture(rhi::Device& device, rhi::TextureHandle texture, rhi::Access current);

// Copies a device-local buffer into memory and waits for the GPU. `current` is the buffer's access
// when called; it is left in the same state. The Null backend has no contents to copy.
std::vector<u8> ReadbackBuffer(rhi::Device& device, rhi::BufferHandle buffer, rhi::Access current);

// Creates a device-local buffer (desc.memory is ignored) holding `data`, ready for shader reads
// (Access::SRVCompute).
rhi::BufferHandle CreateBufferWithData(rhi::Device& device, rhi::BufferDesc desc, const void* data, u64 size);

// Creates a texture (2D or 3D, one mip) holding `texels`, tightly packed rows of desc.width texels
// (x fastest, then y, then the slices of a 3D texture), ready for shader reads (Access::SRVCompute).
rhi::TextureHandle CreateTextureWithData(rhi::Device& device, rhi::TextureDesc desc, const void* texels);

float HalfToFloat(u16 half);
u16 FloatToHalf(float value); // round to nearest even; overflow gives infinity

} // namespace air::renderer
