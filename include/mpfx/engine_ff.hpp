#pragma once

#include <concepts>

#include "types.hpp"

namespace mpfx {

namespace engine_ff {

// Each operation is explicitly instantiated for `float` and `double` in the
// .cpp, keeping the FloppyFloat dependency out of the public headers. `p` must
// not exceed the container type's precision (checked by a debug assertion).

/// @brief Computes `x + y` using FloppyFloat round-to-odd arithmetic.
template <std::floating_point T> T add(T x, T y, prec_t p);

/// @brief Computes `x - y` using FloppyFloat round-to-odd arithmetic.
template <std::floating_point T> T sub(T x, T y, prec_t p);

/// @brief Computes `x * y` using FloppyFloat round-to-odd arithmetic.
template <std::floating_point T> T mul(T x, T y, prec_t p);

/// @brief Computes `x / y` using FloppyFloat round-to-odd arithmetic.
template <std::floating_point T> T div(T x, T y, prec_t p);

/// @brief Computes `sqrt(x)` using FloppyFloat round-to-odd arithmetic.
template <std::floating_point T> T sqrt(T x, prec_t p);

/// @brief Computes `x * y + z` using FloppyFloat round-to-odd arithmetic.
template <std::floating_point T> T fma(T x, T y, T z, prec_t p);

} // end namespace engine_ff

} // end namespace mpfx
