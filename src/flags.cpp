/**
 * @file flags.cpp
 * @brief Status flags based on the IEEE 754 standard.
 * 
 * This file defines the status flags used in the MPFX library,
 * following the IEEE 754 standard for floating-point arithmetic.
 */

#include <mpfx/flags.hpp>

namespace mpfx {

bool invalid = false;
bool div_by_zero = false;
bool overflow = false;
bool tiny_before_rounding = false;
bool tiny_after_rounding = false;
bool underflow_before_rounding = false;
bool underflow_after_rounding = false;
bool inexact = false;
bool carry = false;

} // end namespace mpfx
