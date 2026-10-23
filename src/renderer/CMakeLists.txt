add_library(air_renderer
    src/FeatureRegistry.cpp
    src/FrameRunner.cpp
    src/Parameters.cpp
    src/Scene.cpp
    src/ShaderLoader.cpp
)
target_include_directories(air_renderer PUBLIC include)
target_link_libraries(air_renderer PUBLIC air_rdg)
air_target_defaults(air_renderer)
