#pragma once

#include <bit>
#include <cmath>
#include <concepts>

#include "arch.hpp"
#include "params.hpp"
#include "types.hpp"

namespace mpfx {

namespace engine_fp {

namespace {

template <std::floating_point T>
inline T finalize(T result, unsigned int fexps) {
    using U = typename float_params<T>::uint_t;

    // check if overflow or underflow occurred
    MPFX_DEBUG_ASSERT(
        !(fexps & (arch::EXCEPT_OVERFLOW | arch::EXCEPT_UNDERFLOW)),
        "overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & arch::EXCEPT_INEXACT) {
        U b = std::bit_cast<U>(result);
        b |= 1; // set LSB
        result = std::bit_cast<T>(b);
    }

    return result;
}

} // anonymous namespace

/// @brief Computes `x + y` using round-to-odd arithmetic.
///
/// Requires `p` to not exceed the container type's precision (checked by a
/// debug assertion).
template <std::floating_point T>
inline T add(T x, T y, prec_t p) {
    // the container type only guarantees `P` bits of precision
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "add: requested precision exceeds the container type's capability"
    );

    // prepare floating-point environment
    const auto old_csr = arch::prepare_rto();

    // perform addition in RTZ mode
    const T result = x + y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_csr);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x - y` using round-to-odd arithmetic.
///
/// Requires `p` to not exceed the container type's precision (checked by a
/// debug assertion).
template <std::floating_point T>
inline T sub(T x, T y, prec_t p) {
    // the container type only guarantees `P` bits of precision
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "sub: requested precision exceeds the container type's capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform subtraction in RTZ mode
    const T result = x - y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x * y` using round-to-odd arithmetic.
///
/// Requires `p` to not exceed the container type's precision (checked by a
/// debug assertion).
template <std::floating_point T>
inline T mul(T x, T y, prec_t p) {
    // the container type only guarantees `P` bits of precision
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "mul: requested precision exceeds the container type's capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform multiplication in RTZ mode
    const T result = x * y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x / y` using round-to-odd arithmetic.
///
/// Requires `p` to not exceed the container type's precision (checked by a
/// debug assertion).
template <std::floating_point T>
inline T div(T x, T y, prec_t p) {
    // the container type only guarantees `P` bits of precision
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "div: requested precision exceeds the container type's capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform division in RTZ mode
    const T result = x / y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `sqrt(x)` using round-to-odd arithmetic.
///
/// Requires `p` to not exceed the container type's precision (checked by a
/// debug assertion).
template <std::floating_point T>
inline T sqrt(T x, prec_t p) {
    // the container type only guarantees `P` bits of precision
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "sqrt: requested precision exceeds the container type's capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform square root in RTZ mode
    const T result = std::sqrt(x);

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

/// @brief Computes `x * y + z` using round-to-odd arithmetic.
///
/// Requires `p` to not exceed the container type's precision (checked by a
/// debug assertion).
template <std::floating_point T>
inline T fma(T x, T y, T z, prec_t p) {
    // the container type only guarantees `P` bits of precision
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "fma: requested precision exceeds the container type's capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform fused multiply-add in RTZ mode
    const T result = std::fma(x, y, z);

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

} // end namespace engine_fp

} // end namespace mpfx
