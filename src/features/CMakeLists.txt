# Every subdirectory with a CMakeLists.txt is a feature plugin (see cmake/AirFeatures.cmake).
file(GLOB children LIST_DIRECTORIES true CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(SORT children)
foreach(child IN LISTS children)
    if(IS_DIRECTORY ${child} AND EXISTS ${child}/CMakeLists.txt)
        add_subdirectory(${child})
    endif()
endforeach()
air_generate_builtin_features()
