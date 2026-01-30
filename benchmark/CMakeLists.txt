add_executable(ops ops.cpp)
target_link_libraries(ops
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    ${SOFTFLOAT_LIBRARY}
    floppy_float_static
)
