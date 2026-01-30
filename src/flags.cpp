/**
 * @file flags.cpp
 * @brief Status flags based on the IEEE 754 standard.
 * 
 * This file defines the status flags used in the MPFX library,
 * following the IEEE 754 standard for floating-point arithmetic.
 */

#include <mpfx/flags.hpp>

namespace mpfx {

bool invalid_flag;
bool div_by_zero_flag;
bool overflow_flag;
bool tiny_before_rounding_flag;
bool tiny_after_rounding_flag;
bool underflow_before_rounding_flag;
bool underflow_after_rounding_flag;
bool inexact_flag;
bool carry_flag;

void reset_flags() {
    invalid_flag = false;
    div_by_zero_flag = false;
    overflow_flag = false;
    tiny_before_rounding_flag = false;
    tiny_after_rounding_flag = false;
    underflow_before_rounding_flag = false;
    underflow_after_rounding_flag = false;
    inexact_flag = false;
    carry_flag = false;
}

} // end namespace mpfx
