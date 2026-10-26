# HLSL compilation with DXC at build time.
#
#   air_compile_shaders(TARGET <target> NAME <feature> SOURCES <a.hlsl> ... INCLUDE_DIRS <dir> ...)
#
# Each .hlsl file holds one entry point (docs/coding-standards.md §8). Its kind is detected from the
# source: files with [shader("...")] attributes are ray tracing libraries (lib_6_6), files with
# [numthreads(...)] are compute shaders with entry point Main (cs_6_6). A line
# `// air-shader-target: lib_6_9` (or cs_6_x) sets the kind and profile explicitly, for variants that
# include a shared kernel or need a newer shader model; `// air-shader-dxil-only` skips SPIR-V.
# Shader Model 6.10 targets (cs_6_10: dx/linalg.h) are compiled by the preview dxc
# (AIR_DXC_PREVIEW_EXECUTABLE, with 16-bit types) to DXIL only, and skipped without it.
# Outputs:
#   <build>/shaders/<feature>/<Stem>.lib.dxil | <Stem>.cs.dxil      (always)
#   <build>/shaders/<feature>/<Stem>.lib.spv  | <Stem>.cs.spv       (AIR_RHI_VULKAN only)
# Every .h / .hlsli next to the sources or in an include dir is a dependency, so editing shared
# headers recompiles. Without a dxc (AIR_DXC_EXECUTABLE) this is a no-op.
#
# air_write_shader_manifest() (called once at the end of the top-level CMakeLists.txt) writes
# <build>/shaders/manifest.json: per output, its compiler, arguments and the files it
# depends on. renderer::ShaderHotReload and `tools/air.py shaders` compile from it, with exactly the
# build's command lines.

set(AIR_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(AIR_SHADER_COMMON_INCLUDE "${PROJECT_SOURCE_DIR}/src/renderer/include")

# JSON array of strings from the remaining arguments.
function(air_json_strings outVar)
    set(items "")
    foreach(item IN LISTS ARGN)
        string(REPLACE "\\" "/" item "${item}")
        string(REPLACE "\"" "\\\"" item "${item}")
        list(APPEND items "\"${item}\"")
    endforeach()
    list(JOIN items ", " joined)
    set(${outVar} "[${joined}]" PARENT_SCOPE)
endfunction()

# Records one compiled output for the hot reload manifest. ARGS: compiler arguments without the
# source and -Fo; WATCH: files whose change recompiles it.
function(air_add_shader_manifest_entry)
    cmake_parse_arguments(ARG "" "NAME;SOURCE;OUTPUT;FORMAT;COMPILER" "ARGS;WATCH" ${ARGN})
    air_json_strings(args ${ARG_ARGS})
    air_json_strings(watch ${ARG_WATCH})
    string(REPLACE "\\" "/" compiler "${ARG_COMPILER}")
    set_property(GLOBAL APPEND PROPERTY AIR_SHADER_MANIFEST
        "{\"name\": \"${ARG_NAME}\", \"format\": \"${ARG_FORMAT}\", \"source\": \"${ARG_SOURCE}\", \"output\": \"${ARG_OUTPUT}\", \"compiler\": \"${compiler}\", \"args\": ${args}, \"watch\": ${watch}}")
endfunction()

function(air_write_shader_manifest)
    if(NOT AIR_DXC_EXECUTABLE)
        return()
    endif()
    get_property(entries GLOBAL PROPERTY AIR_SHADER_MANIFEST)
    list(JOIN entries ",\n    " body)
    set(manifest "${AIR_SHADER_OUTPUT_DIR}/manifest.json")
    file(WRITE "${manifest}.in" "{\n  \"compiler\": \"${AIR_DXC_EXECUTABLE}\",\n  \"shaders\": [\n    ${body}\n  ]\n}\n")
    configure_file("${manifest}.in" "${manifest}" COPYONLY) # rewritten only when it changes
endfunction()

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
        if(text MATCHES "// air-shader-target: (lib|cs)_([0-9]+_[0-9]+)")
            set(kind ${CMAKE_MATCH_1})
            set(profile ${CMAKE_MATCH_1}_${CMAKE_MATCH_2})
        elseif(text MATCHES "\\[shader\\(\"")
            set(kind lib)
            set(profile lib_6_6)
        elseif(text MATCHES "\\[numthreads\\(")
            set(kind cs)
            set(profile cs_6_6)
        else()
            message(FATAL_ERROR "${source}: no [shader(...)] or [numthreads(...)] entry point found")
        endif()
        set(compiler "${AIR_DXC_EXECUTABLE}")
        set(previewArgs "")
        set(debugArgs -Qembed_debug -Zi)
        if(profile MATCHES "_6_1[0-9]$")
            if(NOT AIR_DXC_PREVIEW_EXECUTABLE)
                continue() # no preview compiler: the feature falls back to its portable kernels
            endif()
            set(compiler "${AIR_DXC_PREVIEW_EXECUTABLE}")
            set(previewArgs -enable-16bit-types -I "${AIR_DXC_PREVIEW_INCLUDE}")
            # The preview compiler's debug info misplaces inlined functions' scopes, which the
            # driver's DXIL verifier prints at every pipeline creation: these build without it.
            set(debugArgs "")
        endif()
        set(profileArgs -T ${profile})
        if(kind STREQUAL "cs")
            list(APPEND profileArgs -E Main)
        elseif(profile STREQUAL "lib_6_6")
            # Payload access qualifiers are always on from 6.7; on for 6.6 libraries too, so a payload
            # struct compiles the same way for every target.
            list(APPEND profileArgs -enable-payload-qualifiers)
        endif()
        file(GLOB localHeaders CONFIGURE_DEPENDS "${dir}/*.h" "${dir}/*.hlsli")
        # Hot reload watches the source, its directory and the shader headers it can include (not the
        # C++ headers that happen to live in the same include roots).
        set(watch "${source}" ${localHeaders})
        foreach(includeDir IN LISTS ARG_INCLUDE_DIRS)
            file(GLOB extra CONFIGURE_DEPENDS "${includeDir}/*.h" "${includeDir}/*.hlsli")
            list(APPEND watch ${extra})
        endforeach()
        file(GLOB commonShaderHeaders CONFIGURE_DEPENDS "${AIR_SHADER_COMMON_INCLUDE}/air/renderer/shader/*.h")
        list(APPEND watch ${commonShaderHeaders})
        list(REMOVE_DUPLICATES watch)

        set(common ${profileArgs} -HV 2021 -WX -Wno-ignored-attributes -Zpr ${previewArgs} ${includeArgs} -I "${dir}")
        set(dxil "${outDir}/${stem}.${kind}.dxil")
        add_custom_command(
            OUTPUT "${dxil}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${outDir}"
            COMMAND "${compiler}" ${common} ${debugArgs} "${source}" -Fo "${dxil}"
            DEPENDS "${source}" ${localHeaders} ${depends}
            COMMENT "dxc ${ARG_NAME}/${stem}.${kind}.dxil"
            VERBATIM)
        list(APPEND outputs "${dxil}")
        air_add_shader_manifest_entry(NAME ${ARG_NAME} SOURCE "${source}" OUTPUT "${dxil}" FORMAT dxil
                                      COMPILER "${compiler}" ARGS ${common} ${debugArgs} WATCH ${watch})

        if(AIR_RHI_VULKAN AND NOT text MATCHES "// air-shader-dxil-only" AND NOT previewArgs)
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
            air_add_shader_manifest_entry(NAME ${ARG_NAME} SOURCE "${source}" OUTPUT "${spv}" FORMAT spirv
                                          COMPILER "${AIR_DXC_EXECUTABLE}"
                                          ARGS ${common} -spirv -fspv-target-env=vulkan1.3 -fvk-use-scalar-layout
                                          WATCH ${watch})
        endif()
    endforeach()

    add_custom_target(${ARG_TARGET}_shaders DEPENDS ${outputs})
    add_dependencies(${ARG_TARGET} ${ARG_TARGET}_shaders)
endfunction()
