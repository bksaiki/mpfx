#include "mpfx/engine_sf.hpp"

#include <cmath>

extern "C" {
#include <softfloat.h>
}

namespace mpfx {

namespace engine_sf {

/// @brief Computes `x + y` using softfloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
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

/// @brief Computes `x - y` using softfloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "sub: requested precision exceeds double-precision capability"
    );

    // Convert inputs to SoftFloat f64
    float64_t sf_x, sf_y;
    sf_x.v = std::bit_cast<uint64_t>(x);
    sf_y.v = std::bit_cast<uint64_t>(y);

    // Perform subtraction using SoftFloat with round-to-odd mode
    softfloat_roundingMode = softfloat_round_odd;
    float64_t sf_result = f64_sub(sf_x, sf_y);

    // Convert result back to double
    return std::bit_cast<double>(sf_result.v);
}

/// @brief Computes `x * y` using softfloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
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

/// @brief Computes `x / y` using softfloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double div(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "div: requested precision exceeds double-precision capability"
    );

    // Convert inputs to SoftFloat f64
    float64_t sf_x, sf_y;
    sf_x.v = std::bit_cast<uint64_t>(x);
    sf_y.v = std::bit_cast<uint64_t>(y);

    // Perform division using SoftFloat with round-to-odd mode
    softfloat_roundingMode = softfloat_round_odd;
    float64_t sf_result = f64_div(sf_x, sf_y);

    // Convert result back to double
    return std::bit_cast<double>(sf_result.v);
}

/// @brief Computes `sqrt(x)` using softfloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double sqrt(double x, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "sqrt: requested precision exceeds double-precision capability"
    );

    // Convert input to SoftFloat f64
    float64_t sf_x;
    sf_x.v = std::bit_cast<uint64_t>(x);

    // Perform square root using SoftFloat with round-to-odd mode
    softfloat_roundingMode = softfloat_round_odd;
    float64_t sf_result = f64_sqrt(sf_x);

    // Convert result back to double
    return std::bit_cast<double>(sf_result.v);
}

/// @brief Computes `x * y + z` using softfloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double fma(double x, double y, double z, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "fma: requested precision exceeds double-precision capability"
    );

    // Convert inputs to SoftFloat f64
    float64_t sf_x, sf_y, sf_z;
    sf_x.v = std::bit_cast<uint64_t>(x);
    sf_y.v = std::bit_cast<uint64_t>(y);
    sf_z.v = std::bit_cast<uint64_t>(z);

    // Perform fused multiply-add using SoftFloat with round-to-odd mode
    softfloat_roundingMode = softfloat_round_odd;
    float64_t sf_result = f64_mulAdd(sf_x, sf_y, sf_z);

    // Convert result back to double
    return std::bit_cast<double>(sf_result.v);
}

} // end namespace engine_sf

} // end namespace mpfx
