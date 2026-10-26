# Third-party SDKs.
#
# Everything is downloaded at configure time into AIR_DEPS_DIR (default: <source>/.deps, shared by
# all build directories) from pinned versions, or taken from a local copy through the override
# variables below. Nothing is committed to the repository.
#
#   SDK                         Version         Override               Used by
#   DirectX 12 Agility SDK      1.619.6         AIR_AGILITY_SDK_DIR    D3D12 backend (headers + D3D12Core.dll)
#     preview (AIR_D3D12_PREVIEW) 1.721.3-preview                      the same, with SM 6.10 linear algebra
#   DirectX Shader Compiler     1.9.2609.5      AIR_DXC_EXECUTABLE     HLSL -> DXIL / SPIR-V at build time
#     preview (AIR_D3D12_PREVIEW) 1.10.2605.37-preview AIR_DXC_PREVIEW_EXECUTABLE  cs_6_10 shaders only
#   Dear ImGui (docking)        1.93.0          AIR_IMGUI_DIR          editor
#   NVIDIA DLSS SDK             310.9.1         AIR_DLSS_SDK_DIR       dlss feature (Ray Reconstruction)
#   NVIDIA NRD (+ ShaderMake, MathLib) 4.18.0   AIR_NRD_DIR            nrd feature (denoisers), built from source
#   Microsoft WebView2 SDK      1.0.4314.35     AIR_WEBVIEW2_SDK_DIR   editor's HTML profiler panel (Windows)
#
# Targets / functions defined here:
#   air_agility_sdk               INTERFACE: Agility SDK headers (must win over the Windows SDK ones)
#   air_imgui_core                STATIC: ImGui core (editor UI library, any platform)
#   air_imgui                     STATIC: Win32 + DX12 backends (editor executable, Windows)
#   air_dlss                      INTERFACE: NGX headers + import library
#   air_webview2                  INTERFACE: WebView2 headers + static loader (the runtime ships with Windows)
#   air_target_d3d12_runtime(t)   exports the Agility SDK version from executable t and copies the runtime DLLs
#   air_target_dlss_runtime(t)    copies the DLSS DLLs next to executable t

set(AIR_DEPS_DIR "${PROJECT_SOURCE_DIR}/.deps" CACHE PATH "Download cache for third-party SDKs")

set(AIR_AGILITY_SDK_VERSION 1.619.6)
set(AIR_AGILITY_SDK_SHA256 08f0489281401aa430fc37322d6c3fc98a8025175aacd714c10d562f4963f1e9)
set(AIR_DXC_VERSION 1.9.2609.5)
set(AIR_DXC_SHA256 73cfa082245918c2cc1ec6740250eb7377e83a757b3b3a3e16e0d71535bfbfc3)
# Shader Model 6.10 (dx/linalg.h) is preview only: the preview runtime, an experimental feature that
# needs Windows Developer Mode, and the preview compiler for the shaders that use it.
set(AIR_AGILITY_SDK_PREVIEW_VERSION 1.721.3-preview)
set(AIR_AGILITY_SDK_PREVIEW_SHA256 0131bce1e4bace3fc08c03018c29a09ede2570b263721c6449b0ec75762ab22d)
set(AIR_DXC_PREVIEW_VERSION 1.10.2605.37-preview)
set(AIR_DXC_PREVIEW_SHA256 8cf5ec1a8abb832de43251bae1ffcec89b22240677c599c38d30a18dee2f4de6)
set(AIR_IMGUI_TAG v1.93.0-docking)
set(AIR_DLSS_TAG v310.9.1)
set(AIR_NRD_TAG v4.18.0)
set(AIR_SHADERMAKE_COMMIT 3f01623a76b092ac127d2debd79a672f55d77665) # NRD v4.18.0's
set(AIR_MATHLIB_TAG v11)                                             # NRD v4.18.0's
set(AIR_WEBVIEW2_VERSION 1.0.4314.35)
set(AIR_WEBVIEW2_SHA256 90778deb1b77e36292828b4b5833c9255c27533f40e0ddd55891ae718944edc4)

# Downloads url to dest unless it already exists. Sets <outVar> to TRUE on success.
function(air_download url dest outVar)
    set(${outVar} TRUE PARENT_SCOPE)
    if(EXISTS "${dest}")
        return()
    endif()
    set(hashArgs "")
    if(ARGC GREATER 3)
        set(hashArgs EXPECTED_HASH SHA256=${ARGV3})
    endif()
    file(DOWNLOAD "${url}" "${dest}.part" STATUS status TLS_VERIFY ON ${hashArgs})
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        list(GET status 1 message)
        file(REMOVE "${dest}.part")
        message(WARNING "Airender: download failed: ${url} (${message})")
        set(${outVar} FALSE PARENT_SCOPE)
        return()
    endif()
    file(RENAME "${dest}.part" "${dest}")
endfunction()

# Downloads and unpacks a NuGet package. Sets <outVar> to its directory, or "" on failure.
function(air_fetch_nuget id version sha256 outVar)
    string(TOLOWER "${id}" lower)
    set(dir "${AIR_DEPS_DIR}/${id}.${version}")
    if(NOT EXISTS "${dir}/.complete")
        file(MAKE_DIRECTORY "${AIR_DEPS_DIR}")
        air_download("https://api.nuget.org/v3-flatcontainer/${lower}/${version}/${lower}.${version}.nupkg"
                     "${dir}.nupkg" ok ${sha256})
        if(NOT ok)
            set(${outVar} "" PARENT_SCOPE)
            return()
        endif()
        file(ARCHIVE_EXTRACT INPUT "${dir}.nupkg" DESTINATION "${dir}")
        file(TOUCH "${dir}/.complete")
    endif()
    set(${outVar} "${dir}" PARENT_SCOPE)
endfunction()

# Downloads individual files of a GitHub repository at a tag. Sets <outVar> to the local root or "".
function(air_fetch_github_files repo tag name outVar)
    set(dir "${AIR_DEPS_DIR}/${name}-${tag}")
    if(NOT EXISTS "${dir}/.complete")
        foreach(file IN LISTS ARGN)
            air_download("https://raw.githubusercontent.com/${repo}/${tag}/${file}" "${dir}/${file}" ok)
            if(NOT ok)
                set(${outVar} "" PARENT_SCOPE)
                return()
            endif()
        endforeach()
        file(TOUCH "${dir}/.complete")
    endif()
    set(${outVar} "${dir}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# DirectX 12 Agility SDK
# ---------------------------------------------------------------------------
if(AIR_RHI_D3D12)
    set(AIR_AGILITY_SDK_DIR "" CACHE PATH "Local Agility SDK (unpacked Microsoft.Direct3D.D3D12 package)")
    # Only a directory outside the download cache overrides the download (earlier configures cached the
    # downloaded one here, which must follow AIR_D3D12_PREVIEW).
    set(agilityDir "${AIR_AGILITY_SDK_DIR}")
    string(FIND "${agilityDir}" "${AIR_DEPS_DIR}/" inDeps)
    if(NOT agilityDir OR inDeps EQUAL 0)
        if(AIR_D3D12_PREVIEW)
            air_fetch_nuget(Microsoft.Direct3D.D3D12 ${AIR_AGILITY_SDK_PREVIEW_VERSION}
                            ${AIR_AGILITY_SDK_PREVIEW_SHA256} agilityDir)
        else()
            air_fetch_nuget(Microsoft.Direct3D.D3D12 ${AIR_AGILITY_SDK_VERSION} ${AIR_AGILITY_SDK_SHA256} agilityDir)
        endif()
        if(NOT agilityDir)
            message(FATAL_ERROR "Airender: could not download the Agility SDK; set AIR_AGILITY_SDK_DIR")
        endif()
        set(AIR_AGILITY_SDK_DIR "${agilityDir}" CACHE PATH "" FORCE)
    endif()
    message(STATUS "Airender: Agility SDK ${AIR_AGILITY_SDK_DIR}")
    add_library(air_agility_sdk INTERFACE)
    target_include_directories(air_agility_sdk BEFORE INTERFACE "${AIR_AGILITY_SDK_DIR}/build/native/include")

    # The Agility SDK is found through two symbols exported from the executable (not a DLL).
    set(AIR_AGILITY_EXPORTS "${CMAKE_BINARY_DIR}/generated/D3D12AgilityExports.cpp")
    file(WRITE "${AIR_AGILITY_EXPORTS}.in" [[
// Generated by cmake/AirThirdParty.cmake. Selects the DirectX 12 Agility SDK runtime in .\D3D12\.
#include <windows.h>
#include <d3d12.h>
extern "C"
{
#ifdef D3D12_PREVIEW_SDK_VERSION // a preview runtime announces its own version
__declspec(dllexport) extern const UINT D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION;
#else
__declspec(dllexport) extern const UINT D3D12SDKVersion = D3D12_SDK_VERSION;
#endif
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}
]])
    configure_file("${AIR_AGILITY_EXPORTS}.in" "${AIR_AGILITY_EXPORTS}" COPYONLY)

    function(air_target_d3d12_runtime target)
        target_sources(${target} PRIVATE "${AIR_AGILITY_EXPORTS}")
        target_link_libraries(${target} PRIVATE air_agility_sdk)
        set(bin "${AIR_AGILITY_SDK_DIR}/build/native/bin/x64")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/D3D12"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${bin}/D3D12Core.dll" "${bin}/d3d12SDKLayers.dll"
                    "$<TARGET_FILE_DIR:${target}>/D3D12/"
            VERBATIM)
    endfunction()
else()
    function(air_target_d3d12_runtime target)
    endfunction()
endif()

# ---------------------------------------------------------------------------
# DirectX Shader Compiler
# ---------------------------------------------------------------------------
set(AIR_DXC_EXECUTABLE "" CACHE FILEPATH "dxc used to compile HLSL (empty: download on Windows, PATH elsewhere)")
if(NOT AIR_DXC_EXECUTABLE)
    if(CMAKE_HOST_WIN32)
        air_fetch_nuget(Microsoft.Direct3D.DXC ${AIR_DXC_VERSION} ${AIR_DXC_SHA256} dxcDir)
        if(dxcDir)
            set(AIR_DXC_EXECUTABLE "${dxcDir}/build/native/bin/x64/dxc.exe" CACHE FILEPATH "" FORCE)
        endif()
    else()
        find_program(dxcPath NAMES dxc dxc-3.7 HINTS "$ENV{VULKAN_SDK}/bin")
        if(dxcPath)
            set(AIR_DXC_EXECUTABLE "${dxcPath}" CACHE FILEPATH "" FORCE)
        endif()
    endif()
endif()
if(AIR_DXC_EXECUTABLE)
    message(STATUS "Airender: shaders compiled with ${AIR_DXC_EXECUTABLE}")
else()
    message(STATUS "Airender: dxc not found, shaders are not compiled (the Null backend does not need them)")
endif()

# The preview compiler, for the shaders that target Shader Model 6.10 (`// air-shader-target: cs_6_10`);
# without it they are not built and the features that use them fall back to their portable kernels.
set(AIR_DXC_PREVIEW_EXECUTABLE "" CACHE FILEPATH "dxc for Shader Model 6.10 shaders (empty: download with AIR_D3D12_PREVIEW)")
set(AIR_DXC_PREVIEW_INCLUDE "" CACHE PATH "Directory holding dx/linalg.h for the preview dxc")
if(AIR_D3D12_PREVIEW AND NOT AIR_DXC_PREVIEW_EXECUTABLE AND CMAKE_HOST_WIN32)
    air_fetch_nuget(Microsoft.Direct3D.DXC ${AIR_DXC_PREVIEW_VERSION} ${AIR_DXC_PREVIEW_SHA256} dxcPreviewDir)
    if(dxcPreviewDir)
        set(AIR_DXC_PREVIEW_EXECUTABLE "${dxcPreviewDir}/build/native/bin/x64/dxc.exe" CACHE FILEPATH "" FORCE)
        set(AIR_DXC_PREVIEW_INCLUDE "${dxcPreviewDir}/build/native/include/hlsl" CACHE PATH "" FORCE)
    endif()
endif()
if(NOT AIR_D3D12_PREVIEW OR NOT AIR_RHI_D3D12)
    set(AIR_DXC_PREVIEW_EXECUTABLE "") # the shaders it compiles need the preview runtime
endif()
if(AIR_DXC_PREVIEW_EXECUTABLE)
    message(STATUS "Airender: Shader Model 6.10 shaders compiled with ${AIR_DXC_PREVIEW_EXECUTABLE}")
endif()

# ---------------------------------------------------------------------------
# Dear ImGui (docking branch)
# ---------------------------------------------------------------------------
if(AIR_BUILD_EDITOR)
    set(AIR_IMGUI_DIR "" CACHE PATH "Local Dear ImGui checkout (docking branch)")
    if(NOT AIR_IMGUI_DIR)
        air_fetch_github_files(ocornut/imgui ${AIR_IMGUI_TAG} imgui imguiDir
            imgui.h imgui.cpp imgui_internal.h imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp imgui_demo.cpp
            imconfig.h imstb_rectpack.h imstb_textedit.h imstb_truetype.h LICENSE.txt
            backends/imgui_impl_dx12.h backends/imgui_impl_dx12.cpp
            backends/imgui_impl_win32.h backends/imgui_impl_win32.cpp)
        if(imguiDir)
            set(AIR_IMGUI_DIR "${imguiDir}" CACHE PATH "" FORCE)
        endif()
    endif()
    if(NOT AIR_IMGUI_DIR)
        message(WARNING "Airender: could not download Dear ImGui (set AIR_IMGUI_DIR); building without the editor")
        set(AIR_BUILD_EDITOR OFF)
    endif()
endif()
if(AIR_BUILD_EDITOR)
    # Core only: the editor UI library and its headless tests build on every platform.
    add_library(air_imgui_core STATIC
        ${AIR_IMGUI_DIR}/imgui.cpp
        ${AIR_IMGUI_DIR}/imgui_draw.cpp
        ${AIR_IMGUI_DIR}/imgui_tables.cpp
        ${AIR_IMGUI_DIR}/imgui_widgets.cpp
        ${AIR_IMGUI_DIR}/imgui_demo.cpp)
    target_include_directories(air_imgui_core SYSTEM PUBLIC ${AIR_IMGUI_DIR} ${AIR_IMGUI_DIR}/backends)
    target_compile_definitions(air_imgui_core PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
    if(AIR_RHI_D3D12)
        add_library(air_imgui STATIC
            ${AIR_IMGUI_DIR}/backends/imgui_impl_dx12.cpp
            ${AIR_IMGUI_DIR}/backends/imgui_impl_win32.cpp)
        target_link_libraries(air_imgui PUBLIC air_imgui_core air_agility_sdk d3d12 dxgi d3dcompiler dwmapi)
    endif()
endif()

# ---------------------------------------------------------------------------
# Microsoft WebView2 SDK: the editor shows the HTML profile report in a panel. Only the static
# loader is linked; the browser runtime itself is part of Windows 11.
# ---------------------------------------------------------------------------
if(AIR_BUILD_EDITOR AND AIR_RHI_D3D12)
    set(AIR_WEBVIEW2_SDK_DIR "" CACHE PATH "Local WebView2 SDK (unpacked Microsoft.Web.WebView2 package)")
    if(NOT AIR_WEBVIEW2_SDK_DIR)
        air_fetch_nuget(Microsoft.Web.WebView2 ${AIR_WEBVIEW2_VERSION} ${AIR_WEBVIEW2_SHA256} webviewDir)
        if(webviewDir)
            set(AIR_WEBVIEW2_SDK_DIR "${webviewDir}" CACHE PATH "" FORCE)
        endif()
    endif()
    if(AIR_WEBVIEW2_SDK_DIR AND EXISTS "${AIR_WEBVIEW2_SDK_DIR}/build/native/include/WebView2.h")
        add_library(air_webview2 INTERFACE)
        target_include_directories(air_webview2 SYSTEM INTERFACE "${AIR_WEBVIEW2_SDK_DIR}/build/native/include")
        target_link_libraries(air_webview2 INTERFACE "${AIR_WEBVIEW2_SDK_DIR}/build/native/x64/WebView2LoaderStatic.lib"
                                                     version)
    else()
        message(WARNING "Airender: WebView2 SDK not available; the editor opens profile reports in the browser")
    endif()
endif()

# ---------------------------------------------------------------------------
# NVIDIA DLSS SDK (Ray Reconstruction)
# ---------------------------------------------------------------------------
if(AIR_WITH_DLSS)
    set(AIR_DLSS_SDK_DIR "" CACHE PATH "Local NVIDIA DLSS SDK (github.com/NVIDIA/DLSS)")
    if(NOT AIR_DLSS_SDK_DIR)
        set(headers nvsdk_ngx.h nvsdk_ngx_defs.h nvsdk_ngx_params.h nvsdk_ngx_helpers.h
                    nvsdk_ngx_helpers_d3d.h nvsdk_ngx_helpers_cuda.h nvsdk_ngx_loader.h
                    nvsdk_ngx_defs_dlssd.h nvsdk_ngx_params_dlssd.h nvsdk_ngx_helpers_dlssd.h
                    nvsdk_ngx_helpers_dlssd_d3d.h)
        list(TRANSFORM headers PREPEND include/)
        air_fetch_github_files(NVIDIA/DLSS ${AIR_DLSS_TAG} dlss dlssDir
            ${headers} LICENSE.txt
            lib/Windows_x86_64/x64/nvsdk_ngx_d.lib
            lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib
            lib/Windows_x86_64/rel/nvngx_dlss.dll
            lib/Windows_x86_64/rel/nvngx_dlssd.dll)
        if(dlssDir)
            set(AIR_DLSS_SDK_DIR "${dlssDir}" CACHE PATH "" FORCE)
        endif()
    endif()
    if(AIR_DLSS_SDK_DIR AND EXISTS "${AIR_DLSS_SDK_DIR}/include/nvsdk_ngx_helpers_dlssd_d3d.h")
        add_library(air_dlss INTERFACE)
        target_include_directories(air_dlss SYSTEM INTERFACE "${AIR_DLSS_SDK_DIR}/include")
        set(ngxLib "${AIR_DLSS_SDK_DIR}/lib/Windows_x86_64/x64")
        target_link_libraries(air_dlss INTERFACE
            $<IF:$<CONFIG:Debug>,${ngxLib}/nvsdk_ngx_d_dbg.lib,${ngxLib}/nvsdk_ngx_d.lib>)
        message(STATUS "Airender: DLSS SDK ${AIR_DLSS_SDK_DIR}")
    else()
        message(WARNING "Airender: DLSS SDK not available, building without DLSS")
        set(AIR_WITH_DLSS OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# NVIDIA Real-time Denoisers (NRD), built from source with its shaders compiled to DXIL by ShaderMake
# with our dxc. Its own dependencies (ShaderMake, MathLib) are downloaded here too, at the commits its
# CMakeLists.txt names, and handed to its FetchContent declarations.
# ---------------------------------------------------------------------------
# Downloads a GitHub archive (a zip of one top-level folder) and extracts it. Sets <outVar> to that
# folder or "".
function(air_fetch_github_archive url name outVar)
    set(dir "${AIR_DEPS_DIR}/${name}")
    if(NOT EXISTS "${dir}/.complete")
        file(MAKE_DIRECTORY "${AIR_DEPS_DIR}")
        air_download("${url}" "${dir}.zip" ok)
        if(NOT ok)
            set(${outVar} "" PARENT_SCOPE)
            return()
        endif()
        file(REMOVE_RECURSE "${dir}")
        file(ARCHIVE_EXTRACT INPUT "${dir}.zip" DESTINATION "${dir}")
        file(TOUCH "${dir}/.complete")
    endif()
    file(GLOB top LIST_DIRECTORIES true "${dir}/*")
    list(FILTER top EXCLUDE REGEX "\\.complete$")
    list(GET top 0 root)
    set(${outVar} "${root}" PARENT_SCOPE)
endfunction()

if(AIR_WITH_NRD)
    set(AIR_NRD_DIR "" CACHE PATH "Local NRD source (github.com/NVIDIA-RTX/NRD)")
    set(nrdDir "${AIR_NRD_DIR}")
    if(NOT nrdDir)
        air_fetch_github_archive("https://github.com/NVIDIA-RTX/NRD/archive/refs/tags/${AIR_NRD_TAG}.zip"
                                 nrd-${AIR_NRD_TAG} nrdDir)
    endif()
    air_fetch_github_archive(
        "https://github.com/NVIDIA-RTX/ShaderMake/archive/${AIR_SHADERMAKE_COMMIT}.zip"
        shadermake-${AIR_SHADERMAKE_COMMIT} shaderMakeDir)
    air_fetch_github_archive("https://github.com/NVIDIA-RTX/MathLib/archive/refs/tags/${AIR_MATHLIB_TAG}.zip"
                             mathlib-${AIR_MATHLIB_TAG} mathLibDir)
    if(nrdDir AND shaderMakeDir AND mathLibDir AND AIR_DXC_EXECUTABLE AND EXISTS "${nrdDir}/Include/NRD.h")
        set(FETCHCONTENT_SOURCE_DIR_SHADERMAKE "${shaderMakeDir}" CACHE PATH "" FORCE)
        set(FETCHCONTENT_SOURCE_DIR_MATHLIB "${mathLibDir}" CACHE PATH "" FORCE)
        set(SHADERMAKE_FIND_COMPILERS OFF CACHE BOOL "" FORCE)
        set(SHADERMAKE_DXC_PATH "${AIR_DXC_EXECUTABLE}" CACHE INTERNAL "")
        set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
        set(NRD_EMBEDS_DXIL_SHADERS ON CACHE BOOL "" FORCE)
        set(NRD_EMBEDS_SPIRV_SHADERS OFF CACHE BOOL "" FORCE)
        set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
        # NRD's recommended encoding (octahedral R10G10B10A2, linear roughness): the nrd feature packs the
        # path tracer's guide into it (the RGBA16 encodings do not compile with this dxc).
        set(NRD_NORMAL_ENCODING 2 CACHE STRING "" FORCE)
        set(NRD_ROUGHNESS_ENCODING 1 CACHE STRING "" FORCE)
        set(NRD_SHADERS_PATH "${CMAKE_BINARY_DIR}/nrd/Shaders" CACHE STRING "" FORCE)
        add_subdirectory("${nrdDir}" "${CMAKE_BINARY_DIR}/nrd" EXCLUDE_FROM_ALL SYSTEM)
        set(AIR_NRD_SHADER_INCLUDE "${nrdDir}/Shaders" CACHE INTERNAL "") # NRD.hlsli, for our own passes
        message(STATUS "Airender: NRD ${nrdDir}")
    else()
        message(WARNING "Airender: NRD not available, building without the nrd feature")
        set(AIR_WITH_NRD OFF CACHE BOOL "" FORCE)
    endif()
endif()

function(air_target_dlss_runtime target)
    if(TARGET air_dlss)
        set(dlls "${AIR_DLSS_SDK_DIR}/lib/Windows_x86_64/rel")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dlls}/nvngx_dlss.dll" "${dlls}/nvngx_dlssd.dll"
                    "$<TARGET_FILE_DIR:${target}>/"
            VERBATIM)
    endif()
endfunction()
