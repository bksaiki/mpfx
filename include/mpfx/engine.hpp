#pragma once

#include <bit>
#include <cmath>

#include "arch.hpp"
#include "convert.hpp"
#include "types.hpp"

namespace mpfx {

namespace engine {

namespace {

inline double finalize(double result, unsigned int fexps) {
    // check if overflow or underflow occurred
    FPY_DEBUG_ASSERT(
        !(fexps & (arch::EXCEPT_OVERFLOW | arch::EXCEPT_UNDERFLOW)),
        "rto_add: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & arch::EXCEPT_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

} // anonymous namespace

/// @brief Computes `x + y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "rto_add: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_csr = arch::prepare_rto();

    // perform addition in RTZ mode
    double result = x + y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_csr);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x - y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "sub: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform subtraction in RTZ mode
    double result = x - y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
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

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform multiplication in RTZ mode
    double result = x * y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x / y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double div(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "div: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform division in RTZ mode
    double result = x / y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `sqrt(x)` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double sqrt(double x, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "sqrt: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform square root in RTZ mode
    double result = std::sqrt(x);

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x * y + z` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double fma(double x, double y, double z, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 53,
        "fma: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform fused multiply-add in RTZ mode
    double result = std::fma(x, y, z);

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}


/// @brief Computes `x + y` assuming the computation can
/// be done exactly.
///
/// Ensures the result has at least `p` bits of precision.
/// An exception is thrown if the computation is inexact
/// when compiled with FPY_DEBUG.
inline double add_exact(double x, double y, prec_t p) {
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
inline double sub_exact(double x, double y, prec_t p) {
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
inline double mul_exact(double x, double y, prec_t p) {
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


/// @brief Computes `x * y` using fixed-point arithmetic.
/// Returns a fixed-point representation `m * 2^exp`
/// where `m` is an `int64_t` integer significand
/// and `exp` is a base-2 exponent.
inline std::tuple<int64_t, exp_t> mul_fixed(double x, double y, prec_t p) {
    // fixed-point only guarantees 63 bits of precision
    FPY_DEBUG_ASSERT(
        p <= 63,
        "mul_fixed: requested precision exceeds capability"
    );

    // decode into fixed-point
    const auto [xm, xexp] = to_fixed(x);
    const auto [ym, yexp] = to_fixed(y);

    // perform multiplication (possible overflow)
    const int64_t m = xm * ym;
    const exp_t exp = xexp + yexp;

    // return result
    return std::make_tuple(m, exp);
}

} // end namespace engine

} // end namespace mpfx
