air_add_feature(NAME pathtracer
    SOURCES
        src/Bsdf.cpp
        src/PathTracer.cpp
        src/PathTracerFeature.cpp
        src/ReferenceRenderer.cpp
    TESTS
        tests/BsdfTests.cpp
        tests/IntegratorTests.cpp
        tests/LightTreeTests.cpp
        tests/PathTracerTests.cpp
)
