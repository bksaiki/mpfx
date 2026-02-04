#pragma once

#include <bit>

#include "convert.hpp"
#include "types.hpp"

namespace mpfx {

namespace engine_fx {

namespace {

/// @brief Minimizes the precision of `c * 2^exp`.
inline void minimize(mant_t& c, exp_t& exp) {
    const auto tz = std::countr_zero(c);
    c >>= tz;
    exp += static_cast<exp_t>(tz);
}

} // end anonymous namespace

/// @brief Computes `x * y` using fixed-point arithmetic.
/// Returns a fixed-point representation `m * 2^exp`
/// where `m` is an `int64_t` integer significand
/// and `exp` is a base-2 exponent.
inline std::tuple<int64_t, exp_t> mul(double x, double y, prec_t p) {
    // fixed-point only guarantees 63 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 63,
        "mul_fixed: requested precision exceeds capability"
    );

    // decode inputs
    auto [xs, xexp, xc] = unpack_float<double>(x);
    auto [ys, yexp, yc] = unpack_float<double>(y);

    // minimize significands
    minimize(xc, xexp);
    minimize(yc, yexp);

    // apply sign
    const int64_t xm = xs ? -static_cast<int64_t>(xc) : static_cast<int64_t>(xc);
    const int64_t ym = ys ? -static_cast<int64_t>(yc) : static_cast<int64_t>(yc);

    // perform multiplication (possible overflow)
    const int64_t m = xm * ym;
    const exp_t exp = xexp + yexp;

    // return result
    return std::make_tuple(m, exp);
}

} // end namespace engine_fpe

} // end namespace mpfx
