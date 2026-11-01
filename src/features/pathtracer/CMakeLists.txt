air_add_feature(NAME pathtracer
    SOURCES
        src/Bsdf.cpp
        src/CloudMarcher.cpp
        src/PathTracer.cpp
        src/NeuralCoat.cpp
        src/PathTracerFeature.cpp
        src/RadianceCache.cpp
        src/ReferenceRenderer.cpp
        src/ReferenceScene.cpp
        src/ReSTIR.cpp
        src/SubsurfaceFit.cpp
        src/VolumeMarcher.cpp
        src/VolumeMarchReference.cpp
        src/VolumeShadowLayout.cpp
    TESTS
        tests/BsdfTests.cpp
        tests/CoatTests.cpp
        tests/IntegratorTests.cpp
        tests/LightTreeTests.cpp
        tests/PathTracerTests.cpp
        tests/SubsurfaceTests.cpp
        tests/VolumeMarchTests.cpp
    # LibNTC's inference shaders, for neural texture sets decoded on sample (SceneAccess.hlsli).
    SHADER_INCLUDE_DIRS
        ${AIR_NTC_SHADER_INCLUDE}
)
