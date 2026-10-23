add_library(air_pathtracer
    src/PathTracer.cpp
)
target_include_directories(air_pathtracer
    PUBLIC include
    PRIVATE ${PROJECT_SOURCE_DIR}/shaders  # PathTracerShared.h is shared with HLSL
)
target_link_libraries(air_pathtracer PUBLIC air_rdg)
air_target_defaults(air_pathtracer)
