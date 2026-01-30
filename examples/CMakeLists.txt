# Examples

add_executable(benchmark_add benchmark_add.cpp)
target_link_libraries(benchmark_add 
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
)

add_executable(benchmark_mul benchmark_mul.cpp)
target_link_libraries(benchmark_mul 
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
)

add_executable(benchmark_div benchmark_div.cpp)
target_link_libraries(benchmark_div 
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
)

add_executable(benchmark_fma benchmark_fma.cpp)
target_link_libraries(benchmark_fma 
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
)

add_executable(benchmark_sqrt benchmark_sqrt.cpp)
target_link_libraries(benchmark_sqrt 
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
)

add_executable(benchmark_add_engines benchmark_add_engines.cpp)
target_link_libraries(benchmark_add_engines
    mpfx
    softfloat
    floppy_float_static)

add_executable(benchmark_mul_engines benchmark_mul_engines.cpp)
target_link_libraries(benchmark_mul_engines
    mpfx
    softfloat
    floppy_float_static)

add_executable(benchmark_div_engines benchmark_div_engines.cpp)
target_link_libraries(benchmark_div_engines
    mpfx
    softfloat
    floppy_float_static)

add_executable(mixed_dot_prod mixed_dot_prod.cpp)
target_link_libraries(mixed_dot_prod mpfx)
