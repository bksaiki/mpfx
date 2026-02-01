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
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_BENCHMARKS "Build benchmark programs" OFF)
option(ENABLE_PROFILING "Enable profiling support" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -pedantic -DMPFX_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")

# Profiling flags
if(ENABLE_PROFILING)
    # Apply profiling flags globally to all targets
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
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

# FetchContent for external dependencies
include(FetchContent)

# Add GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0 # Use a specific, stable release tag
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# FloppyFloat
set(FLOPPYFLOAT_BUILD_TESTS OFF CACHE BOOL "Disable FloppyFloat tests" FORCE)
set(FLOPPYFLOAT_BUILD_EXAMPLES OFF CACHE BOOL "Disable FloppyFloat examples" FORCE)
FetchContent_Declare(
  floppyfloat
  GIT_REPOSITORY https://github.com/bksaiki/FloppyFloat.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(floppyfloat)

# Include FloppyFloat headers
include_directories(${floppyfloat_SOURCE_DIR}/src)
add_definitions(-DUSE_FLOPPYFLOAT)

# SoftFloat
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(SoftFloat)
if(NOT SOFTFLOAT_FOUND)
    message(FATAL_ERROR "SoftFloat not found")
endif()

include_directories(${SOFTFLOAT_INCLUDE_DIRS})
add_definitions(-DUSE_SOFTFLOAT)

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
    include/mpfx/engine_eft.hpp
    include/mpfx/engine_ff.hpp
    include/mpfx/engine_fp.hpp
    include/mpfx/engine_fpe.hpp
    include/mpfx/engine_fx.hpp
    include/mpfx/engine_sf.hpp
    include/mpfx/flags.hpp
    include/mpfx/ops.hpp
    include/mpfx/params.hpp
    include/mpfx/round.hpp
    include/mpfx/types.hpp
)

# Source files for compiled library
set(SOURCES
    src/context.cpp
    src/context_mpb.cpp
    src/engine_ff.cpp
    src/engine_sf.cpp
    src/flags.cpp
)

# Create compiled library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# Set library version
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories for building and installing
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link against FloppyFloat static library
target_link_libraries(${PROJECT_NAME} PRIVATE floppy_float_static)

# Link against SoftFloat static library
target_link_libraries(${PROJECT_NAME} PRIVATE softfloat)

# Link against GMP and MPFR (needed for other parts of the library)
target_link_libraries(${PROJECT_NAME} PUBLIC ${GMP_LIBRARY} ${MPFR_LIBRARY})

# Install targets
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
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

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
