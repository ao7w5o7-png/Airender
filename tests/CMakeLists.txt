get_property(feature_tests GLOBAL PROPERTY AIR_FEATURE_TESTS)

add_executable(air_tests
    TestMain.cpp
    CoreTests.cpp
    TaskSystemTests.cpp
    GpuMemoryTests.cpp
    RenderGraphTests.cpp
    RendererTests.cpp
    GoldenTests.cpp
    ${feature_tests}
)
target_include_directories(air_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(air_tests PRIVATE AIR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(air_tests PRIVATE air_builtin_features)
air_target_defaults(air_tests)
air_target_d3d12_runtime(air_tests)
add_test(NAME air_tests COMMAND air_tests)
