# HLSL compilation with DXC at build time.
#
#   air_compile_shaders(TARGET <target> NAME <feature> SOURCES <a.hlsl> ... INCLUDE_DIRS <dir> ...)
#
# Each .hlsl file holds one entry point (docs/coding-standards.md §8). Its kind is detected from the
# source: files with [shader("...")] attributes are ray tracing libraries (lib_6_6), files with
# [numthreads(...)] are compute shaders with entry point Main (cs_6_6). Outputs:
#   <build>/shaders/<feature>/<Stem>.lib.dxil | <Stem>.cs.dxil      (always)
#   <build>/shaders/<feature>/<Stem>.lib.spv  | <Stem>.cs.spv       (AIR_RHI_VULKAN only)
# Every .h / .hlsli next to the sources or in an include dir is a dependency, so editing shared
# headers recompiles. Without a dxc (AIR_DXC_EXECUTABLE) this is a no-op.

set(AIR_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(AIR_SHADER_COMMON_INCLUDE "${PROJECT_SOURCE_DIR}/src/renderer/include")

function(air_compile_shaders)
    cmake_parse_arguments(ARG "" "TARGET;NAME" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT AIR_DXC_EXECUTABLE OR NOT ARG_SOURCES)
        return()
    endif()
    set(outDir "${AIR_SHADER_OUTPUT_DIR}/${ARG_NAME}")
    set(includeDirs ${ARG_INCLUDE_DIRS} ${AIR_SHADER_COMMON_INCLUDE})

    set(depends "")
    foreach(dir IN LISTS includeDirs)
        file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${dir}/*.h" "${dir}/*.hlsli")
        list(APPEND depends ${headers})
    endforeach()
    set(includeArgs "")
    foreach(dir IN LISTS includeDirs)
        list(APPEND includeArgs -I "${dir}")
    endforeach()

    set(outputs "")
    foreach(source IN LISTS ARG_SOURCES)
        get_filename_component(stem "${source}" NAME_WE)
        get_filename_component(dir "${source}" DIRECTORY)
        file(READ "${source}" text)
        if(text MATCHES "\\[shader\\(\"")
            set(kind lib)
            set(profileArgs -T lib_6_6)
        elseif(text MATCHES "\\[numthreads\\(")
            set(kind cs)
            set(profileArgs -T cs_6_6 -E Main)
        else()
            message(FATAL_ERROR "${source}: no [shader(...)] or [numthreads(...)] entry point found")
        endif()
        file(GLOB localHeaders CONFIGURE_DEPENDS "${dir}/*.h" "${dir}/*.hlsli")

        set(common ${profileArgs} -HV 2021 -WX -Wno-ignored-attributes -Zpr ${includeArgs} -I "${dir}")
        set(dxil "${outDir}/${stem}.${kind}.dxil")
        add_custom_command(
            OUTPUT "${dxil}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${outDir}"
            COMMAND "${AIR_DXC_EXECUTABLE}" ${common} -Qembed_debug -Zi "${source}" -Fo "${dxil}"
            DEPENDS "${source}" ${localHeaders} ${depends}
            COMMENT "dxc ${ARG_NAME}/${stem}.${kind}.dxil"
            VERBATIM)
        list(APPEND outputs "${dxil}")

        if(AIR_RHI_VULKAN)
            set(spv "${outDir}/${stem}.${kind}.spv")
            add_custom_command(
                OUTPUT "${spv}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${outDir}"
                COMMAND "${AIR_DXC_EXECUTABLE}" ${common} -spirv -fspv-target-env=vulkan1.3 -fvk-use-scalar-layout
                        "${source}" -Fo "${spv}"
                DEPENDS "${source}" ${localHeaders} ${depends}
                COMMENT "dxc ${ARG_NAME}/${stem}.${kind}.spv"
                VERBATIM)
            list(APPEND outputs "${spv}")
        endif()
    endforeach()

    add_custom_target(${ARG_TARGET}_shaders DEPENDS ${outputs})
    add_dependencies(${ARG_TARGET} ${ARG_TARGET}_shaders)
endfunction()
