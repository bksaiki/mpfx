#pragma once

#include <cmath>

extern "C" {
#include <softfloat.h>
}

#include "types.hpp"

namespace mpfx {

namespace engine_sf {

/// @brief Computes `x + y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "add: requested precision exceeds double-precision capability"
    );

    // Convert inputs to SoftFloat f64
    float64_t sf_x, sf_y;
    sf_x.v = std::bit_cast<uint64_t>(x);
    sf_y.v = std::bit_cast<uint64_t>(y);

    // Perform multiplication using SoftFloat with round-to-odd mode
    softfloat_roundingMode = softfloat_round_odd;
    float64_t sf_result = f64_add(sf_x, sf_y);

    // Convert result back to double
    return std::bit_cast<double>(sf_result.v);
}

/// @brief Computes `x * y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "mul: requested precision exceeds double-precision capability"
    );

    // Convert inputs to SoftFloat f64
    float64_t sf_x, sf_y;
    sf_x.v = std::bit_cast<uint64_t>(x);
    sf_y.v = std::bit_cast<uint64_t>(y);

    // Perform multiplication using SoftFloat with round-to-odd mode
    softfloat_roundingMode = softfloat_round_odd;
    float64_t sf_result = f64_mul(sf_x, sf_y);

    // Convert result back to double
    return std::bit_cast<double>(sf_result.v);
}

} // end namespace engine_sf

} // end namespace mpfx
