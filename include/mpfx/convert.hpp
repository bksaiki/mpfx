#pragma once

#include <cmath>
#include <optional>
#include <tuple>

#include "params.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief Converts a double-precision floating-point number `x`
/// to a fixed-point representation `m * 2^exp`,
/// where `m` is an `int64_t` integer significand
/// and `exp` is a base-2 exponent.
///
/// @param x double-precision floating-point number
/// @return a tuple `(m, exp)` representing `x` as `m * 2^exp`
inline std::tuple<int64_t, exp_t> to_fixed(double x) {
    using FP = ieee754_consts<11, 64>; // double precision
    MPFX_ASSERT(std::isfinite(x), "to_fixed: input must be finite");

    // fast path: zero
    if (x == 0.0) {
        return std::make_tuple(0, FP::EXPMIN);
    }

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const bool s = (b >> (FP::N - 1)) != 0;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decode (unsigned) floating-point data
    exp_t exp; // unnormalized exponent
    mant_t c;  // unsigned integer significand
    if (ebits == 0) {
        // subnormal
        exp = FP::EXPMIN;
        c = mbits;
    } else {
        // normal (assuming no infinity or NaN)
        const exp_t e = static_cast<exp_t>(ebits) - FP::BIAS;
        exp = e - static_cast<exp_t>(FP::M);
        c = FP::IMPLICIT1 | mbits;
    }

    // minimize the precision of the significand
    const auto tz = std::countr_zero(c);
    c >>= tz;
    exp += static_cast<exp_t>(tz);

    // apply sign
    const int64_t m = s ? -static_cast<int64_t>(c) : static_cast<int64_t>(c);

    return std::make_tuple(m, exp);
}

/// @brief Constructs a double-precision floating-point number
/// from a digit representation `(-1)^s * c * 2^exp`, where
/// `s` is the sign bit, `c` is the integer significand,
/// and `exp` is the base-2 exponent.
inline double digits_to_double(bool s, exp_t exp, mant_t c) {
    using FP = ieee754_consts<11, 64>;

    // special case: zero
    if (c == 0) {
        return s ? -0.0 : 0.0;
    }

    // requesting maximum precision
    exp_t shift = static_cast<exp_t>(FP::P) - static_cast<exp_t>(std::bit_width(c));
    exp -= shift;
    if (exp < FP::EXPMIN) {
        // requesting lower bound on digits
        // exponent is too small, adjust if necessary
        const exp_t expmin = FP::EXPMIN;
        const exp_t adjust = expmin - exp;
        shift -= adjust;
        exp += adjust;
    }

    // compute new significand `c`
    if (shift > 0) {
        // shift left by a non-negative amount
        const prec_t p_new = std::bit_width(c) + static_cast<prec_t>(shift);
        MPFX_DEBUG_ASSERT(p_new <= 64, "normalize: precision exceeds 64 bits");
        c <<= shift;
    } else if (shift < 0) {
        // shift right by a positive amount
        const mant_t c_lost = c & bitmask<mant_t>(-shift);
        MPFX_DEBUG_ASSERT(c_lost == 0, "normalize: losing digits");
        c >>= -shift;
    }

    // case split on normalization
    const prec_t p = std::bit_width(c);
    const uint64_t ebits = p == FP::P ? (exp - FP::EXPMIN + 1) : 0;
    const uint64_t mbits = c & bitmask<mant_t>(FP::M); 

    // cast to double
    const uint64_t b = (ebits << FP::M) | mbits;
    const double r = std::bit_cast<double>(b);
    return s ? -r : r;
}

} // end namespace mpfx
