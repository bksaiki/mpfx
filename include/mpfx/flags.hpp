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
extern bool invalid_flag;

/// @brief Division by zero flag.
//// An operation produced an exact infinite result for finite operands.
extern bool div_by_zero_flag;

/// @brief Overflow flag.
/// An operation would have produced a value that was larger in magnitude
/// than the largest representable finite value were the exponent
/// range unbounded.
extern bool overflow_flag;

/// @brief Tiny (before rounding) flag.
/// An operation would have produced a value that was smaller in magnitude
/// than the smallest representable normalized value without rounding.
extern bool tiny_before_rounding_flag;

/// @brief Tiny (after rounding) flag.
/// An operation would have produced a value that was smaller in magnitude
/// than the smallest representable normalized value were the exponent
/// range unbounded.
extern bool tiny_after_rounding_flag;

/// @brief Underflow (before rounding) flag.
/// An operation was tiny (before rounding) and inexact.
extern bool underflow_before_rounding_flag;

/// @brief Underflow (after rounding) flag.
/// An operation was tiny (after rounding) and inexact.
extern bool underflow_after_rounding_flag;

/// @brief Inexact flag.
/// An operation produced a result that is different than the numerical
/// result if both precision and exponent range were unbounded.
extern bool inexact_flag;

/// @brief Carry flag.
/// An operation produced a result with a different (normalized) exponent
/// than the result if the precision were unbounded.
extern bool carry_flag;

/// @brief Resets all status flags to false.
void reset_flags();

} // end namespace mpfx
