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

# Link with main library and testing framework
target_link_libraries(run_tests
    mpfx
    GTest::gtest_main
)

# Register tests with CTest
add_test(NAME AllTests COMMAND run_tests)
