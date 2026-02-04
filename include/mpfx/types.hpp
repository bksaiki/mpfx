#pragma once

#include <cstdint>

#include "utils.hpp"

namespace mpfx {

/// @brief floating-point exponent
using exp_t = int32_t;

/// @brief container type for a floating-point mantissa
using mant_t = uint64_t;

/// @brief container type for precision
using prec_t = uint64_t;

/// @brief Generates a bitmask of length `k` for type `T`.
/// @tparam T an unsigned integral type 
/// @param k size of the bitmask
/// @return the bitmask
template <typename T>
constexpr T bitmask(uint64_t k) {
    MPFX_STATIC_ASSERT(std::is_integral_v<T> && std::is_unsigned_v<T>, "T must be an unsigned integer");
    constexpr uint64_t MAX_K = 8 * sizeof(T);
    return k < MAX_K ? static_cast<T>((static_cast<T>(1) << k) - 1) : ~static_cast<T>(0);
}

} // namespace mpfx
