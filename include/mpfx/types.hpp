#pragma once

#include <concepts>
#include <cstdint>

#include "utils.hpp"

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

template <unsigned_integral T>
constexpr int bit_width(T x) {
    if constexpr (std::is_same_v<T, uint128_t>) {
        // compute bit width of 128-bit integer using two 64-bit widths
        const uint64_t low = static_cast<uint64_t>(x);
        const uint64_t high = static_cast<uint64_t>(x >> 64);
        return high ? 64 + std::bit_width(high) : std::bit_width(low);
    } else {
        return std::bit_width(x);
    }
}

} // namespace mpfx
