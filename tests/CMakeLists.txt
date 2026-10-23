get_property(feature_tests GLOBAL PROPERTY AIR_FEATURE_TESTS)

add_executable(air_tests
    TestMain.cpp
    CoreTests.cpp
    TaskSystemTests.cpp
    GpuMemoryTests.cpp
    RenderGraphTests.cpp
    RendererTests.cpp
    GoldenTests.cpp
    DiagnosticsTests.cpp
    ProfileReportTests.cpp
    ShaderHotReloadTests.cpp
    SceneTests.cpp
    SkyTests.cpp
    ${feature_tests}
)
target_include_directories(air_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(air_tests PRIVATE AIR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
                                             AIR_BUILD_SHADER_DIR="${AIR_SHADER_OUTPUT_DIR}")
target_link_libraries(air_tests PRIVATE air_builtin_features)
air_target_defaults(air_tests)
air_target_d3d12_runtime(air_tests)
add_test(NAME air_tests COMMAND air_tests)

# The editor UI runs headless on the Null device (no window, no ImGui backends).
if(TARGET air_editor_ui)
    target_sources(air_tests PRIVATE EditorTests.cpp)
    target_link_libraries(air_tests PRIVATE air_editor_ui)
endif()
