#pragma once

#include <bit>

#include "convert.hpp"
#include "types.hpp"

namespace mpfx {

namespace engine_fpe {

/// @brief Computes `x + y` assuming the computation can
/// be done exactly.
///
/// Ensures the result has at least `p` bits of precision.
/// An exception is thrown if the computation is inexact
/// when compiled with FPY_DEBUG.
inline double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "add_exact: requested precision exceeds double-precision capability"
    );

#if defined(FPY_DEBUG)
    // prepare floating-point environment
    arch::clear_exceptions();
#endif

    // perform exact addition
    double result = x + y;

#if defined(FPY_DEBUG)
    // check for inexactness or overflow
    const auto fexps = arch::get_exceptions();
    FPY_DEBUG_ASSERT(
        !(fexps & (arch::EXCEPT_INEXACT | arch::EXCEPT_OVERFLOW)),
        "add_exact: addition was not exact"
    );
#endif

    // return result
    return result;
}

/// @brief Computes `x - y` assuming the computation can
/// be done exactly.
///
/// Ensures the result has at least `p` bits of precision.
/// An exception is thrown if the computation is inexact
/// when compiled with FPY_DEBUG.
inline double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "sub_exact: requested precision exceeds double-precision capability"
    );

#if defined(FPY_DEBUG)
    // prepare floating-point environment
    arch::clear_exceptions();
#endif

    // perform exact subtraction
    double result = x - y;

#if defined(FPY_DEBUG)
    // check for inexactness or overflow
    const auto fexps = arch::get_exceptions();
    FPY_DEBUG_ASSERT(
        !(fexps & (arch::EXCEPT_INEXACT | arch::EXCEPT_OVERFLOW)),
        "sub_exact: subtraction was not exact"
    );
#endif

    // return result
    return result;
}

/// @brief Computes `x * y` assuming the computation can
/// be done exactly.
///
/// Ensures the result has at least `p` bits of precision.
/// An exception is thrown if the computation is inexact
/// when compiled with FPY_DEBUG.
inline double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "mul_exact: requested precision exceeds double-precision capability"
    );

#if defined(FPY_DEBUG)
    // prepare floating-point environment
    arch::clear_exceptions();
#endif

    // perform exact multiplication
    double result = x * y;

#if defined(FPY_DEBUG)
    // check for inexactness or overflow
    const auto fexps = arch::get_exceptions();
    FPY_DEBUG_ASSERT(
        !(fexps & (arch::EXCEPT_INEXACT | arch::EXCEPT_OVERFLOW)),
        "mul_exact: multiplication was not exact"
    );
#endif

    // return result
    return result;
}

} // end namespace engine_fpe

} // end namespace mpfx
