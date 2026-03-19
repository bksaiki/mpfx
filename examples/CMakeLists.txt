# Examples

add_executable(mixed_dot_prod mixed_dot_prod.cpp)
target_link_libraries(mixed_dot_prod mpfx)

add_executable(mx_dot_prod mx_dot_prod.cpp)
target_link_libraries(mx_dot_prod mpfx softfloat)
