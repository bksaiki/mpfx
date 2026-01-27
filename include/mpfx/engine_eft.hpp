/**
 * @file engine_eft.hpp
 * @brief Engine implementation using error-free transformations (EFTs).
 * 
 * This file provides functions for basic arithmetic operations
 * using EFTs to compute a round-to-odd result. Since floating-point
 * status flags are slow to use, EFTs might be faster.
 */

#pragma once

#include <cmath>
#include <concepts>
#include <tuple>

#include "types.hpp"

namespace mpfx {

namespace engine_eft {

namespace {

/// @brief Type trait to give an unsigned integer type of the same size
/// as a floating-point type.
template <std::floating_point T>
struct float_to_uint;

template <>
struct float_to_uint<float> {
    using type = uint32_t;
};

template <>
struct float_to_uint<double> {
    using type = uint64_t;
};


/// @brief Gives the next representable floating-point value towards zero.
///
/// Assumes `x` is finite and non-zero.
template <std::floating_point T>
inline T next_toward_zero(const T& x) {
    using U = typename float_to_uint<T>::type;
    auto u = std::bit_cast<U>(x);
    u -= x > 0. ? 1 : -1;
    return std::bit_cast<T>(u);
}

/// @brief Finalizes the rounding of an EFT result to round-to-odd.
///
/// Assumes `high` and `low` are both finite.
template <std::floating_point T>
inline T round_finalize(T high, T low) {
    MPFX_DEBUG_ASSERT(std::isfinite(high), "round_finalize: high part is not finite");
    MPFX_DEBUG_ASSERT(std::isfinite(low), "round_finalize: low part is not finite");
    using U = typename float_to_uint<T>::type;

    if (low == 0.0) {
        // result is exact
        return high;
    } else {
        // result is inexact
        // `high` and `low` are both non-zero

        // make sure `high` is the RTZ result
        if (std::signbit(high) == std::signbit(low)) {
            // `high` is not the RTZ result, so adjust it by
            // "borrowing" from the low part
            high = next_toward_zero(high);
        }

        // apply RTZ rounding
        U b = std::bit_cast<U>(high);
        b |= 1; // set LSB to make odd
        return std::bit_cast<T>(b);
    }
}

template <std::floating_point T>
inline std::tuple<T, T> two_sum(const T& x, const T& y) {
    const bool no_swap = std::fabs(x) > std::fabs(y);
    const T s = x + y;
    const T a = no_swap ? x : y;
    const T b = no_swap ? y : x;
    const T t = (s - a) - b;
    return { s, t };
}

template <std::floating_point T>
inline std::tuple<T, T> two_prod(const T& x, const T& y) {
    const T p = x * y;
    const T e = std::fma(x, y, -p);
    return { p, e };
}

} // end anonymous namespace


/// @brief Computes `x + y` using error-free transformation.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "add: requested precision exceeds double-precision capability"
    );

    if (!std::isfinite(x) || !std::isfinite(y)) {
        // handle special values using standard addition
        return x + y;
    }

    // perform EFT addition
    const auto [s, t] = two_sum(x, y);

    // finalize rounding to round-to-odd
    return round_finalize(s, t);
}

/// @brief Computes `x - y` using error-free transformation.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "sub: requested precision exceeds double-precision capability"
    ); 

    if (!std::isfinite(x) || !std::isfinite(y)) {
        // handle special values using standard subtraction
        return x - y;
    }

    // perform EFT subtraction
    const auto [s, t] = two_sum(x, -y);

    // finalize rounding to round-to-odd
    return round_finalize(s, t);
}

/// @brief Computes `x * y` using error-free transformation.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "mul: requested precision exceeds double-precision capability"
    );

    if (!std::isfinite(x) || !std::isfinite(y)) {
        // handle special values using standard multiplication
        return x * y;
    }

    // perform EFT multiplication
    const auto [s, t] = two_prod(x, y);

    // finalize rounding to round-to-odd
    return round_finalize(s, t);
}

} // end namespace engine_eft

} // end namespace mpfx
