air_add_feature(NAME pathtracer
    SOURCES
        src/Bsdf.cpp
        src/PathTracer.cpp
        src/PathTracerFeature.cpp
    TESTS
        tests/BsdfTests.cpp
        tests/PathTracerTests.cpp
)
