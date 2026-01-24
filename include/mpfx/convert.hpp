#pragma once

#include <tuple>

#include "types.hpp"

namespace mpfx {

/// @brief Converts a double-precision floating-point number `x`
/// to a fixed-point representation `m * 2^exp`,
/// where `m` is an `int64_t` integer significand
/// and `exp` is a base-2 exponent.
///
/// @param x double-precision floating-point number
/// @return a tuple `(m, exp)` representing `x` as `m * 2^exp`
std::tuple<int64_t, exp_t> to_fixed(double x);

/// @brief Constructs a double-precision floating-point number
/// from a digit representation `(-1)^s * c * 2^exp`, where
/// `s` is the sign bit, `c` is the integer significand,
/// and `exp` is the base-2 exponent.
double digits_to_double(bool s, exp_t exp, mant_t c);

} // end namespace mpfx
