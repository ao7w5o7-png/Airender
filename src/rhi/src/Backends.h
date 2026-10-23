#pragma once

#include "air/rhi/RHIDevice.h"

namespace air::rhi
{

std::unique_ptr<Device> CreateNullDevice(const DeviceDesc& desc);
#if AIR_RHI_D3D12
std::unique_ptr<Device> CreateD3D12Device(const DeviceDesc& desc);
#endif
#if AIR_RHI_VULKAN
std::unique_ptr<Device> CreateVulkanDevice(const DeviceDesc& desc);
#endif

} // namespace air::rhi
