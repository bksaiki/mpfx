add_executable(ops ops.cpp)
target_link_libraries(ops
    mpfx
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
    softfloat
    floppy_float_static)
)
