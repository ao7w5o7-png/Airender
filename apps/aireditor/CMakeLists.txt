# The editor is split in two:
#   air_editor_ui  platform-independent UI and frame logic (Dear ImGui core only). Builds everywhere
#                  and is exercised headless by tests/EditorTests.cpp on the Null backend.
#   aireditor      the Windows executable: Win32 window, DXGI swapchain, ImGui Win32 + DX12 backends.
add_library(air_editor_ui
    EditorApp.cpp
    DiagnosticsPanels.cpp
    Widgets.cpp
)
target_include_directories(air_editor_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(air_editor_ui PUBLIC air_builtin_features air_imgui_core)
target_compile_definitions(air_editor_ui PUBLIC AIR_DEFAULT_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
air_target_defaults(air_editor_ui)

if(AIR_RHI_D3D12)
    add_executable(aireditor WIN32 MainWin32.cpp)
    target_link_libraries(aireditor PRIVATE air_editor_ui air_imgui)
    if(MINGW)
        target_link_options(aireditor PRIVATE -municode) # entry point is wWinMain
    endif()
    air_target_defaults(aireditor)
    air_target_d3d12_runtime(aireditor)
    if(COMMAND air_target_dlss_runtime)
        air_target_dlss_runtime(aireditor)
    endif()
endif()
