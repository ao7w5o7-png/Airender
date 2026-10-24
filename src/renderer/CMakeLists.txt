add_library(air_renderer
    src/BlueNoise.cpp
    src/Clouds.cpp
    src/Diagnostics.cpp
    src/FeatureRegistry.cpp
    src/FrameRunner.cpp
    src/LightTree.cpp
    src/Mlp.cpp
    src/MlpCheck.cpp
    src/NeuralNetwork.cpp
    src/Parameters.cpp
    src/ProfileReport.cpp
    src/Transfer.cpp
    src/Scene.cpp
    src/SceneLibrary.cpp
    src/Sky.cpp
    src/TestScenes.cpp
    src/TextureSampling.cpp
    src/Vdb.cpp
    src/Volume.cpp
    src/VolumeBricks.cpp
    src/ShaderHotReload.cpp
    src/ShaderLoader.cpp
)
target_include_directories(air_renderer PUBLIC include)
target_link_libraries(air_renderer PUBLIC air_rdg)
air_target_defaults(air_renderer)
# The profile report page (ProfileReport.h), compiled in so reports need no files next to the binary.
air_embed_text(TARGET air_renderer SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/report/ProfileReport.html
               NAME ProfileReportHtml)
# The renderer's own shaders (the fallback upscale, the neural networks' optimizer), loaded from
# <shader root>/renderer.
air_compile_shaders(TARGET air_renderer NAME renderer
                    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Upscale.hlsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/MlpOptimize.hlsl
                            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/MlpCheck.hlsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/MlpCheckPortable.hlsl
                    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
