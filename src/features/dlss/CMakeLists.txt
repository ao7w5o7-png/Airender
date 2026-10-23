# NVIDIA DLSS Ray Reconstruction. Needs the DLSS SDK (fetched by cmake/AirThirdParty.cmake on
# Windows with D3D12); elsewhere the feature is simply not built.
if(TARGET air_dlss)
    air_add_feature(NAME dlss
        SOURCES
            src/DlssFeature.cpp
        TESTS
            tests/DlssTests.cpp
        LIBRARIES
            air_dlss
    )
endif()
