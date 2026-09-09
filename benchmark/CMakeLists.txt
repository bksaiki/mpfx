# GMP and MPFR headers are used by the benchmarks as a reference oracle.
include_directories(
    ${GMP_INCLUDE_DIR}
    ${MPFR_INCLUDE_DIR}
)

# Rounding-layer microbenchmark (status-flag overhead).
add_executable(benchmark_round benchmark_round.cpp)
target_link_libraries(benchmark_round
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)

# CPFloat shim. Built without OpenMP so CPFloat runs single-threaded like every
# other baseline; -w silences its warning about the optional PCG dependency.
add_library(cpfloat_wrap STATIC cpfloat_wrap.c)
target_include_directories(cpfloat_wrap
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${cpfloat_SOURCE_DIR}/src
)
target_compile_options(cpfloat_wrap PRIVATE -w)
target_link_libraries(cpfloat_wrap PRIVATE m)

# Comprehensive per-op benchmark: every MPFX engine vs MPFR/SoftFloat/FloppyFloat,
# swept over the f32 and f64 container types (CSV output).
add_executable(benchmark_ops benchmark_ops.cpp)
target_link_libraries(benchmark_ops
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
    floppy_float_static
    cpfloat_wrap
)
