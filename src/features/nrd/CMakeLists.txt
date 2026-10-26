# NVIDIA Real-time Denoisers (NRD), the default denoiser of the path tracer in real time. NRD is built
# from source by cmake/AirThirdParty.cmake (D3D12, AIR_WITH_NRD); elsewhere the feature is not built.
if(TARGET NRD)
    air_add_feature(NAME nrd
        SOURCES
            src/NrdFeature.cpp
        TESTS
            tests/NrdTests.cpp
        LIBRARIES
            NRD
            d3d12
        SHADER_INCLUDE_DIRS
            ${AIR_NRD_SHADER_INCLUDE}
    )
endif()
