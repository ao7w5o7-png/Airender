add_library(air_rdg
    src/RenderGraph.cpp
    src/TransientResourcePool.cpp
)
target_include_directories(air_rdg PUBLIC include)
target_link_libraries(air_rdg PUBLIC air_rhi air_core)
air_target_defaults(air_rdg)
