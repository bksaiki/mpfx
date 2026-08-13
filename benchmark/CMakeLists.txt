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

# Rounding-layer microbenchmark: the scale-and-truncate implementations against
# the integer-domain one, swept over both container types, every rounding mode,
# and four input distributions (CSV output). Needs no reference oracle.
add_executable(benchmark_round_scaled benchmark_round_scaled.cpp)
target_link_libraries(benchmark_round_scaled mpfx)

# Comprehensive per-op benchmark: every MPFX engine vs MPFR/SoftFloat/FloppyFloat,
# swept over the f32 and f64 container types (CSV output).
add_executable(benchmark_ops benchmark_ops.cpp)
target_link_libraries(benchmark_ops
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
    floppy_float_static
)
