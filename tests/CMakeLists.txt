# Tests CMakeLists.txt

# Test executables
set(TEST_SOURCES
    test_bit_float.cpp
    test_convert.cpp
    test_context.cpp
    test_engine.cpp
    test_flags.cpp
    test_ops.cpp
    test_round.cpp
    test_eft.cpp
)

# Create test executable
add_executable(run_tests ${TEST_SOURCES})

# Link with main library and testing framework.
# GMP and MPFR are used by the tests (e.g. test_ops.cpp) as a reference oracle.
target_include_directories(run_tests PRIVATE
    ${GMP_INCLUDE_DIR}
    ${MPFR_INCLUDE_DIR}
)

target_link_libraries(run_tests
    mpfx
    GTest::gtest_main
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)

# Register tests with CTest
add_test(NAME AllTests COMMAND run_tests)
