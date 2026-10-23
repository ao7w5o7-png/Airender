add_executable(air_tests
    TestMain.cpp
    TaskSystemTests.cpp
    GpuMemoryTests.cpp
    RenderGraphTests.cpp
    PathTracerTests.cpp
)
target_link_libraries(air_tests PRIVATE air_pathtracer)
air_target_defaults(air_tests)
add_test(NAME air_tests COMMAND air_tests)
