add_library(air_renderer
    src/Diagnostics.cpp
    src/FeatureRegistry.cpp
    src/FrameRunner.cpp
    src/Parameters.cpp
    src/Transfer.cpp
    src/Scene.cpp
    src/SceneLibrary.cpp
    src/ShaderLoader.cpp
)
target_include_directories(air_renderer PUBLIC include)
target_link_libraries(air_renderer PUBLIC air_rdg)
air_target_defaults(air_renderer)
# FrameRunner's own shaders (the fallback upscale), loaded from <shader root>/renderer.
air_compile_shaders(TARGET air_renderer NAME renderer SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Upscale.hlsl
                    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
