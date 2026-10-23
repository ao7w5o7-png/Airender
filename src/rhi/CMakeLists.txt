add_library(air_rhi
    src/RHI.cpp
    src/GpuMemoryAllocator.cpp
    src/null/NullDevice.cpp
)
target_include_directories(air_rhi PUBLIC include)
target_link_libraries(air_rhi PUBLIC air_core)
air_target_defaults(air_rhi)

if(AIR_RHI_D3D12)
    target_sources(air_rhi PRIVATE
        src/d3d12/D3D12Device.cpp
        src/d3d12/D3D12CommandList.cpp
        src/d3d12/D3D12RayTracing.cpp
        src/d3d12/D3D12Swapchain.cpp
    )
    target_compile_definitions(air_rhi PUBLIC AIR_RHI_D3D12=1 PRIVATE NOMINMAX)
    target_link_libraries(air_rhi PRIVATE air_agility_sdk d3d12 dxgi dxguid)
endif()

if(AIR_RHI_VULKAN)
    find_package(Vulkan REQUIRED)
    target_sources(air_rhi PRIVATE src/vulkan/VulkanDevice.cpp)
    target_compile_definitions(air_rhi PUBLIC AIR_RHI_VULKAN=1)
    target_link_libraries(air_rhi PRIVATE Vulkan::Vulkan)
endif()
