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

/// @brief Finalizes the rounding of an EFT result to round-to-odd.
/// Assumes `high` and `low` are both finite.
template <std::floating_point T>
inline T round_finalize(T high, T low) {
    MPFX_DEBUG_ASSERT(std::isfinite(high), "round_finalize: high part is not finite");
    MPFX_DEBUG_ASSERT(std::isfinite(low), "round_finalize: low part is not finite");
    using FP = typename float_params<T>::params;
    using U = typename float_params<T>::uint_t;
    static constexpr auto SIGN_SHIFT = FP::N - 1;

    // fast path: low part is zero
    if (low == static_cast<T>(0)) {
        return high; // exact result, no adjustment needed
    }

    // slow path: low part is non-zero (so is high part)
    const U b_high = std::bit_cast<U>(high);
    const U b_low = std::bit_cast<U>(low);

    // compute sign difference
    const int sign_high = b_high >> SIGN_SHIFT;
    const int sign_low = b_low >> SIGN_SHIFT;
    const int sign_diff = sign_high ^ sign_low;

    // compute adjustment for RTZ: +1 for negative `high`, -1 for positive `high`
    // only apply if the signs differ
    const int adjust_mask = -static_cast<int>(sign_diff);
    const int adjust = static_cast<int>((sign_high << 1) - 1) & adjust_mask;

    // apply adjustment and jam sticky bit for RTO
    U result = static_cast<U>(b_high + adjust);
    result |= 1;

    // reinterpret back to floating-point
    return std::bit_cast<T>(result);
}

template <std::floating_point T>
inline std::tuple<T, T> two_sum(T x, T y) {
    const bool swap = std::fabs(x) < std::fabs(y);
    const T a = swap ? y : x;
    const T b = swap ? x : y;

    const T s = a + b;
    const T yy = s - a;
    const T t = b - yy;
    return { s, t };
}

template <std::floating_point T>
inline std::tuple<T, T> two_prod(T x, T y) {
    const T p = x * y;
    const T e = std::fma(x, y, -p);
    return { p, e };
}

/// @brief Error-free transformation of division.
///
/// Computes `x / y = q + r` such that `q` is the
/// round-to-nearest result of `x / y`, and `r` is
/// the error term.
template <std::floating_point T>
inline std::tuple<T, T> two_div(T x, T y) {
    const T q = x / y;
    const T r = -std::fma(q, y, -x) / y;
    return { q, r };
}

/// @brief Error-free transformation of square root
///
/// Computes `sqrt(x) = q + r` such that `q` is the
/// round-to-nearest result of `sqrt(x)`, and `r` is
/// the error term.
template <std::floating_point T>
inline std::tuple<T, T> two_sqrt(T x) {
    const T r1 = std::sqrt(x);
    const T n = std::fma(-r1, r1, x);
    const T d = static_cast<T>(2) * r1;
    const T r2 = n / d;
    return { r1, r2 };
}

/// @brief Error-free transformation of FMA.
///
/// Computes `x * y + z = r1 + r2` such that `r1` is the
/// round-to-nearest result of `x * y + z`, and `r2` is
/// the error term.
template <std::floating_point T>
inline std::tuple<T, T> eft_fma(T x, T y, T z) {
    const auto r1 = std::fma(x, y, z);
    const auto [u1, u2] = two_prod(x, y);
    const auto [a1, a2] = two_sum(z, u2);
    const auto [b1, b2] = two_sum(u1, a1);
    const auto g = (b1 - r1) + b2;
    // const auto [r2, r3] = two_sum(g, a2);
    // return { r1, r2, r3 };
    const auto r2 = g + a2;
    return { r1, r2 };
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

    if (!std::isfinite(x) || !std::isfinite(y)) [[unlikely]] {
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

    if (!std::isfinite(x) || !std::isfinite(y)) [[unlikely]] {
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

    if (!std::isfinite(x) || !std::isfinite(y)) [[unlikely]] {
        // handle special values using standard multiplication
        return x * y;
    }

    // perform EFT multiplication
    const auto [s, t] = two_prod(x, y);

    // finalize rounding to round-to-odd
    return round_finalize(s, t);
}

/// @brief Computes `x / y` using an error-free transformation.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double div(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "div: requested precision exceeds double-precision capability"
    );

    if (!std::isfinite(x) || !std::isfinite(y) || y == 0.0) [[unlikely]] {
        // handle special values using standard division
        return x / y;
    }

    // perform EFT division
    const auto [q, t] = two_div(x, y);

    // finalize rounding to round-to-odd
    return round_finalize(q, t);
}

/// @brief Computes `sqrt(x)` using an error-free transformation.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double sqrt(double x, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "sqrt: requested precision exceeds double-precision capability"
    );

    if (!std::isfinite(x) || x <= 0.0) [[unlikely]] {
        // handle special values using standard square root
        return std::sqrt(x);
    }

    // perform EFT square root
    const auto [r1, r2] = two_sqrt(x);

    // finalize the rounding to round-to-odd
    return round_finalize(r1, r2);
}

/// @brief Computes `x * y + z` using an error-free transformation.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double fma(double x, double y, double z, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "fma: requested precision exceeds double-precision capability"
    );

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) [[unlikely]] {
        // handle special values using standard FMA
        return std::fma(x, y, z);
    }

    // perform EFT of FMA
    const auto [r1, r2] = eft_fma(x, y, z);

    // finalize the rounding to round-to-odd
    return round_finalize(r1, r2);
}

} // end namespace engine_eft

} // end namespace mpfx
