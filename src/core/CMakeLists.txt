add_library(air_core
    src/Compression.cpp
    src/Image.cpp
    src/Json.cpp
    src/Log.cpp
    src/Process.cpp
    src/Profiler.cpp
    src/TaskSystem.cpp
)
target_include_directories(air_core PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(air_core PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(air_core PRIVATE psapi)
endif()
air_target_defaults(air_core)
