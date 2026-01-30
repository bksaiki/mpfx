/**
 * @file flags.hpp
 * @brief Status flags based on the IEEE 754 standard.
 * 
 * This file defines the status flags used in the MPFX library,
 * following the IEEE 754 standard for floating-point arithmetic.
 */

#pragma once

namespace mpfx {

/// @brief Invalid operation flag.
/// An operation had no usefully defineable result.
extern bool invalid;

/// @brief Division by zero flag.
//// An operation produced an exact infinite result for finite operands.
extern bool div_by_zero;

/// @brief Overflow flag.
/// An operation would have produced a value that was larger in magnitude
/// than the largest representable finite value were the exponent
/// range unbounded.
extern bool overflow;

/// @brief Tiny (before rounding) flag.
/// An operation would have produced a value that was smaller in magnitude
/// than the smallest representable normalized value without rounding.
extern bool tiny_before_rounding;

/// @brief Tiny (after rounding) flag.
/// An operation would have produced a value that was smaller in magnitude
/// than the smallest representable normalized value were the exponent
/// range unbounded.
extern bool tiny_after_rounding;

/// @brief Underflow (before rounding) flag.
/// An operation was tiny (before rounding) and inexact.
extern bool underflow_before_rounding;

/// @brief Underflow (after rounding) flag.
/// An operation was tiny (after rounding) and inexact.
extern bool underflow_after_rounding;

/// @brief Inexact flag.
/// An operation produced a result that is different than the numerical
/// result if both precision and exponent range were unbounded.
extern bool inexact;

/// @brief Carry flag.
/// An operation produced a result with a different (normalized) exponent
/// than the result if the precision were unbounded.
extern bool carry;

} // end namespace mpfx
