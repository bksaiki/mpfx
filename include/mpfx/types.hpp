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
inline T bitmask(uint64_t k) {
    MPFX_STATIC_ASSERT(std::is_integral_v<T> && std::is_unsigned_v<T>, "T must be an unsigned integer");
    static constexpr uint64_t MAX_K = 8 * sizeof(T);
    MPFX_DEBUG_ASSERT(k < MAX_K, "exceeded maximum bitmask size for T");
    return static_cast<T>((static_cast<T>(1) << k) - 1);
}

} // namespace mpfx
