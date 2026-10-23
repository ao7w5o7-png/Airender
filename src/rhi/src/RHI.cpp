#include "Backends.h"

#include "air/core/Log.h"

namespace air::rhi {

std::unique_ptr<Device> CreateDevice(const DeviceDesc& desc) {
    switch (desc.backend) {
    case Backend::Null: return CreateNullDevice(desc);
    case Backend::D3D12:
#if AIR_RHI_D3D12
        return CreateD3D12Device(desc);
#else
        AIR_LOG(Error, "RHI", "D3D12 backend not compiled in (configure with -DAIR_RHI_D3D12=ON on Windows)");
        return nullptr;
#endif
    case Backend::Vulkan:
#if AIR_RHI_VULKAN
        return CreateVulkanDevice(desc);
#else
        AIR_LOG(Error, "RHI", "Vulkan backend not compiled in (configure with -DAIR_RHI_VULKAN=ON)");
        return nullptr;
#endif
    }
    return nullptr;
}

const char* ToString(Backend backend) {
    switch (backend) {
    case Backend::Null: return "Null";
    case Backend::D3D12: return "D3D12";
    case Backend::Vulkan: return "Vulkan";
    }
    return "?";
}

u32 FormatBytesPerPixel(Format format) {
    switch (format) {
    case Format::Unknown: return 0;
    case Format::R8_UNORM: return 1;
    case Format::RG8_UNORM:
    case Format::R16_FLOAT: return 2;
    case Format::RGBA8_UNORM:
    case Format::RGBA8_SRGB:
    case Format::BGRA8_UNORM:
    case Format::RG16_FLOAT:
    case Format::R32_FLOAT:
    case Format::R32_UINT:
    case Format::R11G11B10_FLOAT:
    case Format::D32_FLOAT:
    case Format::D24_UNORM_S8_UINT: return 4;
    case Format::RGBA16_FLOAT:
    case Format::RG32_FLOAT:
    case Format::RG32_UINT: return 8;
    case Format::RGB32_FLOAT: return 12;
    case Format::RGBA32_FLOAT:
    case Format::RGBA32_UINT: return 16;
    }
    return 0;
}

u32 TextureCopyRowPitch(Format format, u32 width) {
    return u32(AlignUp(u64(width) * FormatBytesPerPixel(format), 256));
}

bool IsDepthFormat(Format format) {
    return format == Format::D32_FLOAT || format == Format::D24_UNORM_S8_UINT;
}

const char* ToString(Format format) {
    switch (format) {
#define AIR_FORMAT_CASE(f)                                                                                             \
    case Format::f: return #f;
        AIR_FORMAT_CASE(Unknown)
        AIR_FORMAT_CASE(R8_UNORM)
        AIR_FORMAT_CASE(RG8_UNORM)
        AIR_FORMAT_CASE(RGBA8_UNORM)
        AIR_FORMAT_CASE(RGBA8_SRGB)
        AIR_FORMAT_CASE(BGRA8_UNORM)
        AIR_FORMAT_CASE(R16_FLOAT)
        AIR_FORMAT_CASE(RG16_FLOAT)
        AIR_FORMAT_CASE(RGBA16_FLOAT)
        AIR_FORMAT_CASE(R32_FLOAT)
        AIR_FORMAT_CASE(RG32_FLOAT)
        AIR_FORMAT_CASE(RGB32_FLOAT)
        AIR_FORMAT_CASE(RGBA32_FLOAT)
        AIR_FORMAT_CASE(R32_UINT)
        AIR_FORMAT_CASE(RG32_UINT)
        AIR_FORMAT_CASE(RGBA32_UINT)
        AIR_FORMAT_CASE(R11G11B10_FLOAT)
        AIR_FORMAT_CASE(D32_FLOAT)
        AIR_FORMAT_CASE(D24_UNORM_S8_UINT)
#undef AIR_FORMAT_CASE
    }
    return "?";
}

const char* ToString(MemoryLocation location) {
    switch (location) {
    case MemoryLocation::DeviceLocal: return "DeviceLocal";
    case MemoryLocation::Upload: return "Upload";
    case MemoryLocation::Readback: return "Readback";
    case MemoryLocation::DeviceLocalHost: return "DeviceLocalHost";
    case MemoryLocation::Count: break;
    }
    return "?";
}

std::string ToString(Access access) {
    if (access == Access::None) {
        return "None";
    }
    static constexpr const char* kNames[] = {"CopySrc",         "CopyDst",          "VertexOrIndex", "Constant",
                                             "SRVGraphics",     "SRVCompute",       "UAVGraphics",   "UAVCompute",
                                             "RenderTarget",    "DepthRead",        "DepthWrite",    "IndirectArgs",
                                             "AccelStructRead", "AccelStructWrite", "Present",       "HostRead"};
    std::string result;
    for (u32 bit = 0; bit < std::size(kNames); ++bit) {
        if (u32(access) & (1u << bit)) {
            if (!result.empty()) {
                result += '|';
            }
            result += kNames[bit];
        }
    }
    return result;
}

} // namespace air::rhi
