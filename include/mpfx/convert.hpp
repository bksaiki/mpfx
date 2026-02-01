#pragma once

#include <cmath>
#include <concepts>
#include <optional>
#include <tuple>

#include "params.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief Constructs a floating-point number from a representation
/// `(-1)^s * c * 2^exp`, where `s` is the sign bit, `exp` is the
/// base-2 exponent, and `c` is the integer significand.
template <std::floating_point T>
inline T make_float(bool s, exp_t exp, typename float_params<T>::uint_t c) {
    using FP = typename float_params<T>::params;
    using uint_t = typename float_params<T>::uint_t;

    // special case: zero
    if (c == 0) {
        return s ? -static_cast<T>(0) : static_cast<T>(0);
    }

    // compute the normalized exponent
    const prec_t p = static_cast<prec_t>(std::bit_width(c));
    exp_t e = exp + (static_cast<exp_t>(p) - 1);

    // normalize `c` to have `FP::P` bits of precision
    const exp_t shift = static_cast<exp_t>(FP::P) - static_cast<exp_t>(p);
    if (shift > 0) {
        // not enough precision, shift left
        c <<= shift;
    } else if (shift < 0) {
        // too much precision, shift right
        const uint_t c_lost = c & bitmask<uint_t>(-shift);
        MPFX_DEBUG_ASSERT(c_lost == 0, "make_float: losing digits due to normalization");
        c >>= -shift;
    }

    uint_t ebits, mbits;
    if (e < FP::EMIN) {
        // subnormal result

        // check if we shifted off any digits
        const exp_t adjust = FP::EMIN - e;
        const uint_t c_lost = c & bitmask<uint_t>(adjust);
        MPFX_DEBUG_ASSERT(c_lost == 0, "make_float: losing digits due to subnormalization");

        ebits = 0;
        mbits = c >> adjust;
    } else {
        // normal result
        ebits = e + FP::BIAS;
        mbits = c & static_cast<uint_t>(FP::MMASK);
    }

    // cast to float
    const uint_t b = (ebits << FP::M) | mbits;
    const T r = std::bit_cast<T>(b);
    return s ? -r : r;
}

/// @brief Unpacks a floating-point number `x` into a triple
/// `(s, exp, c)` where `s` is the sign, `exp` is the (unnormalized)
/// exponent, and `c` is the unsigned integer significand,
/// i.e., `x = (-1)^s * c * 2^exp`.
///
/// Assumes `x` is a finite number (not infinity or NaN).
///
/// @param x floating-point number
/// @return a tuple `(s, exp, c)` representing `x` as `(-1)^s * c * 2^exp`
template <std::floating_point T>
inline std::tuple<bool, exp_t, typename float_params<T>::uint_t> unpack_float(T x) {
    using FP = typename float_params<T>::params;
    using uint_t = typename float_params<T>::uint_t;
    MPFX_DEBUG_ASSERT(std::isfinite(x), "unpack_float: input must be finite");

    // convert to bits
    const uint_t b = std::bit_cast<uint_t>(x);
    const bool s = (b & static_cast<uint_t>(FP::SMASK)) != 0;
    const uint_t ebits = (b & static_cast<uint_t>(FP::EMASK)) >> FP::M;
    const uint_t mbits = b & static_cast<uint_t>(FP::MMASK);

    // unbias the exponent
    exp_t exp;
    uint_t c;
    if (ebits == 0) [[unlikely]] {
        // subnormal
        exp = FP::EXPMIN;
        c = mbits;
    } else [[likely]] {
        // normal (assuming no infinity or NaN)
        const exp_t e = static_cast<exp_t>(ebits) - FP::BIAS;
        exp = e - static_cast<exp_t>(FP::M);
        c = static_cast<uint_t>(FP::IMPLICIT1) | mbits;
    }

    // return the unpacked representation
    return std::make_tuple(s, exp, c);
}


/// @brief Converts a double-precision floating-point number `x`
/// to a fixed-point representation `m * 2^exp`,
/// where `m` is an `int64_t` integer significand
/// and `exp` is a base-2 exponent.
///
/// @param x double-precision floating-point number
/// @return a tuple `(m, exp)` representing `x` as `m * 2^exp`
inline std::tuple<int64_t, exp_t> to_fixed(double x) {
    using FP = float_params<double>::params; // double precision
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

} // end namespace mpfx
