add_library(air_core
    src/Log.cpp
    src/TaskSystem.cpp
)
target_include_directories(air_core PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(air_core PUBLIC Threads::Threads)
air_target_defaults(air_core)
