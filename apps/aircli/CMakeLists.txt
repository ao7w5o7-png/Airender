add_executable(aircli main.cpp)
target_link_libraries(aircli PRIVATE air_pathtracer)
air_target_defaults(aircli)
