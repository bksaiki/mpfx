#pragma once

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include "utils.hpp"

#if defined(__has_builtin)
    #if __has_builtin(__builtin_roundeven)
        #define MPFX_HAS_ROUNDEVEN
    #endif
#endif

namespace mpfx {

#ifdef __SIZEOF_INT128__
using int128_t = __int128;
using uint128_t = unsigned __int128;
#else
#error "int128 is not supported with this compiler"
#endif

/// @brief floating-point exponent
using exp_t = int32_t;

/// @brief container type for a floating-point mantissa
using mant_t = uint64_t;

/// @brief container type for precision
using prec_t = uint64_t;

/// @brief Concept for signed integral types including `int128_t`
/// @tparam T the type to check
template <typename T>
concept signed_integral = std::signed_integral<T> || std::is_same_v<T, int128_t>;

/// @brief Concept for unsigned integral types including `uint128_t`
/// @tparam T the type to check
template <typename T>
concept unsigned_integral = std::unsigned_integral<T> || std::is_same_v<T, uint128_t>;

/// @brief Like `std::make_unsigned<T>` but also supports `uint128_t`.
/// @tparam T a signed integral type
template <typename T>
struct make_unsigned {
    using type = std::make_unsigned<T>::type;
};

template <>
struct make_unsigned<int128_t> {
    using type = uint128_t;
};

/// @brief Like `std::make_unsigned_t<T>` but also supports `uint128_t`.
/// @tparam T a signed integral type
template <typename T>
using make_unsigned_t = make_unsigned<T>::type;

/// @brief Generates a bitmask of length `k` for type `T`.
/// @tparam T an unsigned integral type 
/// @param k size of the bitmask
/// @return the bitmask
template <unsigned_integral T>
constexpr T bitmask(uint64_t k) {
    constexpr uint64_t MAX_K = 8 * sizeof(T);
    return k < MAX_K ? static_cast<T>((static_cast<T>(1) << k) - 1) : ~static_cast<T>(0);
}

/// @brief Like `std::bit_width` but also supports `uint128_t`.
/// @tparam T an unsigned integral type
/// @param x an unsigned integer
/// @return the smallest integer greater than the base-2 logarithm of `x`.
template <unsigned_integral T>
constexpr int bit_width(T x) {
    return std::bit_width(x);
}

template <>
constexpr int bit_width<uint128_t>(uint128_t x) {
    // compute bit width of 128-bit integer using two 64-bit widths
    const uint64_t low = static_cast<uint64_t>(x);
    const uint64_t high = static_cast<uint64_t>(x >> 64);
    return high ? 64 + std::bit_width(high) : std::bit_width(low);
}

/// @brief Rounds to the nearest integral value, ties to even.
/// @tparam T a floating-point type
/// @param x the value to round
/// @return `x` rounded to an integral value
///
/// This is `roundeven` from C23. Unlike `std::nearbyint` and `std::rint`, which
/// round according to the dynamic rounding mode, it is always ties-to-even: on x86
/// and ARM64 a single instruction with the mode pinned (`roundsd $8` / `frintn`),
/// where the standard functions emit the "use the current mode" form instead. The
/// fallback keeps the same semantics for compilers without the builtin; every step
/// of it is exact.
template <std::floating_point T>
inline T round_even(T x) {
#ifdef MPFX_HAS_ROUNDEVEN
    if constexpr (std::same_as<T, float>) {
        return __builtin_roundevenf(x);
    } else if constexpr (std::same_as<T, long double>) {
        return __builtin_roundevenl(x);
    } else {
        return __builtin_roundeven(x);
    }
#else
    const T t = std::trunc(x);
    const T r = std::fabs(x - t);
    if (r < static_cast<T>(0.5)) {
        return t;
    }
    if (r > static_cast<T>(0.5)) {
        return t + std::copysign(static_cast<T>(1), x);
    }

    // exactly halfway - step away from zero only if `t` is odd, which is
    // exactly when halving it leaves a fraction
    const T h = t * static_cast<T>(0.5);
    return h == std::trunc(h) ? t : t + std::copysign(static_cast<T>(1), x);
#endif
}

} // namespace mpfx
