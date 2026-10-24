air_add_feature(NAME pathtracer
    SOURCES
        src/Bsdf.cpp
        src/PathTracer.cpp
        src/PathTracerFeature.cpp
        src/RadianceCache.cpp
        src/ReferenceRenderer.cpp
        src/ReSTIR.cpp
        src/VolumeMarcher.cpp
    TESTS
        tests/BsdfTests.cpp
        tests/IntegratorTests.cpp
        tests/LightTreeTests.cpp
        tests/PathTracerTests.cpp
)
