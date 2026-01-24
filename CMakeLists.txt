cmake_minimum_required(VERSION 3.15)

# NOTE: Update project name, version, and description as needed
project(mpfx
    VERSION 0.1.0
    DESCRIPTION "Number library for software floating-point arithmetic"
    LANGUAGES CXX
)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(ENABLE_PROFILING "Enable profiling support" OFF)
option(USE_SOFTFLOAT "Enable SoftFloat benchmarks" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -pedantic -DFPY_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")

# Profiling flags
if(ENABLE_PROFILING)
    # Apply profiling flags globally to all targets
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer")
endif()

# GMP
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY gmp)

if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR "GMP not found")
endif()

# MPFR
find_path(MPFR_INCLUDE_DIR mpfr.h)
find_library(MPFR_LIBRARY mpfr)

if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY)
    message(FATAL_ERROR "MPFR not found")
endif()

# SoftFloat (optional)
if(USE_SOFTFLOAT)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
    find_package(SoftFloat)
    if(SOFTFLOAT_FOUND)
        message(STATUS "SoftFloat found: ${SOFTFLOAT_LIBRARIES}")
        add_definitions(-DUSE_SOFTFLOAT)
    else()
        message(WARNING "SoftFloat not found. Run install_softfloat.sh to install it.")
        set(USE_SOFTFLOAT OFF)
    endif()
endif()

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0 # Use a specific, stable release tag
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${GMP_INCLUDE_DIR}
    ${MPFR_INCLUDE_DIR}
)

# Header files (for IDE organization)
set(HEADERS
    include/mpfx/context.hpp
    include/mpfx/context_ieee754.hpp
    include/mpfx/context_mp.hpp
    include/mpfx/context_mps.hpp
    include/mpfx/context_mpb.hpp
    include/mpfx/convert.hpp
    include/mpfx/engine.hpp
    include/mpfx/ops.hpp
    include/mpfx/params.hpp
    include/mpfx/round.hpp
    include/mpfx/types.hpp
)

# Create interface library (header-only)
add_library(${PROJECT_NAME} INTERFACE)

# Set interface include directories
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install targets
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
)

install(DIRECTORY include/mpfx
    DESTINATION include
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
