#include "mpfx/engine_sf.hpp"

#include <cmath>

extern "C" {
#include <softfloat.h>
}

namespace mpfx {

namespace engine_sf {

// Converts from double to float64_t
static float64_t to_sf(double x) {
    float64_t result;
    result.v = std::bit_cast<uint64_t>(x);
    return result;
}

// Converts from float64_t to double
static double from_sf(float64_t x) {
    return std::bit_cast<double>(x.v);
}

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

    softfloat_roundingMode = softfloat_round_odd;
    return from_sf(f64_add(to_sf(x), to_sf(y)));
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

    softfloat_roundingMode = softfloat_round_odd;
    return from_sf(f64_sub(to_sf(x), to_sf(y)));
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

    softfloat_roundingMode = softfloat_round_odd;
    return from_sf(f64_mul(to_sf(x), to_sf(y)));
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

    softfloat_roundingMode = softfloat_round_odd;
    return from_sf(f64_div(to_sf(x), to_sf(y)));
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

    softfloat_roundingMode = softfloat_round_odd;
    return from_sf(f64_sqrt(to_sf(x)));
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

    softfloat_roundingMode = softfloat_round_odd;
    return from_sf(f64_mulAdd(to_sf(x), to_sf(y), to_sf(z)));
}

} // end namespace engine_sf

} // end namespace mpfx
