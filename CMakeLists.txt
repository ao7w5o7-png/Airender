cmake_minimum_required(VERSION 3.24)
project(Airender VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
if(WIN32)
    set(AIR_D3D12_DEFAULT ON)
else()
    set(AIR_D3D12_DEFAULT OFF)
endif()

option(AIR_RHI_D3D12  "Build the Direct3D 12 RHI backend" ${AIR_D3D12_DEFAULT})
option(AIR_RHI_VULKAN "Build the Vulkan RHI backend (requires Vulkan SDK)" OFF)
option(AIR_BUILD_TESTS "Build unit tests" ON)
option(AIR_BUILD_APPS  "Build command line apps" ON)
option(AIR_BUILD_EDITOR "Build the ImGui editor (UI library + tests everywhere, executable on Windows/D3D12)" ON)
option(AIR_WITH_DLSS "Build the DLSS feature (Windows, D3D12, NVIDIA DLSS SDK)" ${AIR_D3D12_DEFAULT})
# Preview Agility SDK + DXC: Shader Model 6.10 linear algebra (tensor-core matrix-vector products for
# the neural radiance cache). Off: the retail SDK, and neural networks run their portable kernels.
option(AIR_D3D12_PREVIEW "Use the preview Agility SDK and DXC for Shader Model 6.10 linear algebra" ${AIR_D3D12_DEFAULT})
option(AIR_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(AIR_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if(AIR_RHI_D3D12 AND NOT WIN32)
    message(FATAL_ERROR "AIR_RHI_D3D12 requires Windows")
endif()
if(NOT AIR_RHI_D3D12)
    set(AIR_WITH_DLSS OFF)
endif()

include(cmake/AirCompileOptions.cmake)
include(cmake/AirEmbed.cmake)
include(cmake/AirThirdParty.cmake)
include(cmake/AirShaders.cmake)
include(cmake/AirFeatures.cmake)

add_subdirectory(src/core)
add_subdirectory(src/rhi)
add_subdirectory(src/rdg)
add_subdirectory(src/renderer)
add_subdirectory(src/features)

if(AIR_BUILD_APPS)
    add_subdirectory(apps/aircli)
endif()
if(AIR_BUILD_EDITOR AND EXISTS ${PROJECT_SOURCE_DIR}/apps/aireditor/CMakeLists.txt)
    add_subdirectory(apps/aireditor)
endif()

if(AIR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

air_write_shader_manifest()

message(STATUS "Airender: D3D12=${AIR_RHI_D3D12} Vulkan=${AIR_RHI_VULKAN} Editor=${AIR_BUILD_EDITOR} "
               "DLSS=${AIR_WITH_DLSS} Tests=${AIR_BUILD_TESTS}")
