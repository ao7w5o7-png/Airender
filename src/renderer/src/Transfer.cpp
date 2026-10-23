#include "air/renderer/Transfer.h"

#include "air/core/Log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>

namespace air::renderer
{

using rhi::Access;
using rhi::Barrier;
using rhi::Format;

float HalfToFloat(u16 half)
{
    const u32 sign = u32(half & 0x8000u) << 16;
    const u32 exponent = (half >> 10) & 0x1Fu;
    u32 mantissa = half & 0x3FFu;
    u32 bits = 0;
    if (exponent == 0x1F)
    {
        bits = sign | 0x7F800000u | (mantissa << 13); // inf / nan
    }
    else if (exponent != 0)
    {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    else if (mantissa != 0)
    {
        // Subnormal: normalize.
        u32 e = 113;
        while ((mantissa & 0x400u) == 0)
        {
            mantissa <<= 1;
            --e;
        }
        bits = sign | (e << 23) | ((mantissa & 0x3FFu) << 13);
    }
    else
    {
        bits = sign;
    }
    return std::bit_cast<float>(bits);
}

rhi::BufferHandle CreateBufferWithData(rhi::Device& device, rhi::BufferDesc desc, const void* data, u64 size)
{
    desc.memory = rhi::MemoryLocation::DeviceLocal;
    desc.usage |= rhi::BufferUsage::CopyDst;
    desc.size = std::max(desc.size, size);
    const rhi::BufferHandle buffer = device.CreateBuffer(desc);
    const rhi::BufferHandle staging = device.CreateBuffer({.size = size,
                                                           .usage = rhi::BufferUsage::CopySrc,
                                                           .memory = rhi::MemoryLocation::Upload,
                                                           .debugName = desc.debugName + ".Staging"});
    if (!buffer.IsValid() || !staging.IsValid())
    {
        AIR_LOG(Error, "Transfer", "out of memory creating '{}'", desc.debugName);
        return buffer;
    }
    std::memcpy(device.GetMappedPointer(staging), data, size);
    auto cmd = device.CreateCommandList(rhi::QueueType::Graphics);
    cmd->Begin();
    cmd->Barriers(std::array{Barrier::Buf(buffer, Access::None, Access::CopyDst)});
    cmd->CopyBuffer(buffer, 0, staging, 0, size);
    cmd->Barriers(std::array{Barrier::Buf(buffer, Access::CopyDst, Access::SRVCompute)});
    cmd->End();
    rhi::CommandList* lists[] = {cmd.get()};
    device.WaitForFence(rhi::QueueType::Graphics, device.Submit(rhi::QueueType::Graphics, lists));
    device.DestroyBuffer(staging);
    return buffer;
}

rhi::TextureHandle CreateTextureWithData(rhi::Device& device, rhi::TextureDesc desc, const void* texels)
{
    desc.usage |= rhi::TextureUsage::CopyDst | rhi::TextureUsage::ShaderResource;
    desc.mipLevels = 1;
    const u32 depth = desc.dimension == rhi::TextureDimension::Tex3D ? desc.depthOrArraySize : 1u;
    const u64 rowBytes = u64(desc.width) * rhi::FormatBytesPerPixel(desc.format);
    const u64 pitch = rhi::TextureCopyRowPitch(desc.format, desc.width);
    const u64 rows = u64(desc.height) * depth;
    const rhi::TextureHandle texture = device.CreateTexture(desc);
    const rhi::BufferHandle staging = device.CreateBuffer({.size = pitch * rows,
                                                           .usage = rhi::BufferUsage::CopySrc,
                                                           .memory = rhi::MemoryLocation::Upload,
                                                           .debugName = desc.debugName + ".Staging"});
    if (!texture.IsValid() || !staging.IsValid())
    {
        AIR_LOG(Error, "Transfer", "out of memory creating '{}'", desc.debugName);
        return texture;
    }
    auto* mapped = static_cast<u8*>(device.GetMappedPointer(staging));
    for (u64 row = 0; row < rows; ++row)
    {
        std::memcpy(mapped + row * pitch, static_cast<const u8*>(texels) + row * rowBytes, rowBytes);
    }
    auto cmd = device.CreateCommandList(rhi::QueueType::Graphics);
    cmd->Begin();
    cmd->Barriers(std::array{Barrier::Tex(texture, Access::None, Access::CopyDst)});
    cmd->CopyBufferToTexture(texture, 0, staging, 0);
    cmd->Barriers(std::array{Barrier::Tex(texture, Access::CopyDst, Access::SRVCompute)});
    cmd->End();
    rhi::CommandList* lists[] = {cmd.get()};
    device.WaitForFence(rhi::QueueType::Graphics, device.Submit(rhi::QueueType::Graphics, lists));
    device.DestroyBuffer(staging);
    return texture;
}

std::optional<Image> ReadbackTexture(rhi::Device& device, rhi::TextureHandle texture, Access current)
{
    const rhi::TextureDesc* desc = device.GetTextureDesc(texture);
    if (!desc || (desc->format != Format::RGBA8_UNORM && desc->format != Format::RGBA16_FLOAT &&
                  desc->format != Format::RGBA32_FLOAT))
    {
        AIR_LOG(Error, "Readback", "unsupported texture for readback");
        return std::nullopt;
    }
    const u32 width = desc->width;
    const u32 height = desc->height;
    const Format format = desc->format;
    const u32 rowPitch = rhi::TextureCopyRowPitch(format, width);
    const rhi::BufferHandle buffer = device.CreateBuffer({.size = u64(rowPitch) * height,
                                                          .usage = rhi::BufferUsage::CopyDst,
                                                          .memory = rhi::MemoryLocation::Readback,
                                                          .debugName = "Readback"});
    auto cmd = device.CreateCommandList(rhi::QueueType::Graphics);
    cmd->Begin();
    cmd->Barriers(std::array{Barrier::Tex(texture, current, Access::CopySrc)});
    cmd->CopyTextureToBuffer(buffer, 0, texture, 0);
    cmd->Barriers(std::array{Barrier::Tex(texture, Access::CopySrc, current)});
    cmd->End();
    rhi::CommandList* lists[] = {cmd.get()};
    device.WaitForFence(rhi::QueueType::Graphics, device.Submit(rhi::QueueType::Graphics, lists));

    const auto* bytes = static_cast<const u8*>(device.GetMappedPointer(buffer));
    Image image(width, height);
    for (u32 y = 0; y < height && bytes; ++y)
    {
        const u8* row = bytes + size_t(y) * rowPitch;
        for (u32 x = 0; x < width; ++x)
        {
            float* out = image.At(x, y);
            for (u32 c = 0; c < 4; ++c)
            {
                if (format == Format::RGBA8_UNORM)
                {
                    out[c] = float(row[x * 4 + c]) / 255.0f;
                }
                else if (format == Format::RGBA16_FLOAT)
                {
                    u16 half = 0;
                    std::memcpy(&half, row + (x * 4 + c) * 2, 2);
                    out[c] = HalfToFloat(half);
                }
                else
                {
                    std::memcpy(&out[c], row + (x * 4 + c) * 4, 4);
                }
            }
        }
    }
    device.DestroyBuffer(buffer);
    return image;
}

} // namespace air::renderer
