#pragma once

#include <bit>

#include "convert.hpp"
#include "types.hpp"

namespace mpfx {

namespace engine_fx {

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

    // decode into fixed-point
    const auto [xm, xexp] = to_fixed(x);
    const auto [ym, yexp] = to_fixed(y);

    // perform multiplication (possible overflow)
    const int64_t m = xm * ym;
    const exp_t exp = xexp + yexp;

    // return result
    return std::make_tuple(m, exp);
}

} // end namespace engine_fpe

} // end namespace mpfx
