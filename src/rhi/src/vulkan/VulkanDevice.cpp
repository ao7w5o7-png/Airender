// Vulkan backend (work in progress).
//
// Plan (see docs/architecture.md, "RHI"):
//   * Vulkan 1.3 core: dynamic rendering, synchronization2, timeline semaphores, buffer device address
//   * VK_KHR_acceleration_structure + VK_KHR_ray_tracing_pipeline + VK_KHR_ray_query for the path tracer
//   * VK_EXT_descriptor_indexing (core 1.2) for one global bindless descriptor set
//   * GpuMemoryAllocator with a Vulkan IGpuHeapProvider (VkDeviceMemory per block)
//     memory type key = memoryTypeIndex chosen from vkGetPhysicalDeviceMemoryProperties
//   * Shaders: the same HLSL compiled by DXC with -spirv

#include "../Backends.h"

#include "air/core/Log.h"

namespace air::rhi {

std::unique_ptr<Device> CreateVulkanDevice(const DeviceDesc& desc) {
    AIR_LOG(Error, "Vulkan", "Vulkan backend is not implemented yet");
    return nullptr;
}

} // namespace air::rhi
