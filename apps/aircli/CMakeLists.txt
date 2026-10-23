add_executable(aircli main.cpp)
target_link_libraries(aircli PRIVATE air_builtin_features)
air_target_defaults(aircli)
