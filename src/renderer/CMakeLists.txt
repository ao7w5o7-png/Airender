add_library(air_renderer
    src/Diagnostics.cpp
    src/FeatureRegistry.cpp
    src/FrameRunner.cpp
    src/Parameters.cpp
    src/ProfileReport.cpp
    src/Transfer.cpp
    src/Scene.cpp
    src/SceneLibrary.cpp
    src/TestScenes.cpp
    src/ShaderHotReload.cpp
    src/ShaderLoader.cpp
)
target_include_directories(air_renderer PUBLIC include)
target_link_libraries(air_renderer PUBLIC air_rdg)
air_target_defaults(air_renderer)
# The profile report page (ProfileReport.h), compiled in so reports need no files next to the binary.
air_embed_text(TARGET air_renderer SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/report/ProfileReport.html
               NAME ProfileReportHtml)
# FrameRunner's own shaders (the fallback upscale), loaded from <shader root>/renderer.
air_compile_shaders(TARGET air_renderer NAME renderer SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Upscale.hlsl
                    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
