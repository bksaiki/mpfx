/**
 * @file flags.hpp
 * @brief Status flags based on the IEEE 754 standard.
 * 
 * This file defines the status flags used in the MPFX library,
 * following the IEEE 754 standard for floating-point arithmetic.
 */

#pragma once

#include <cstdint>

namespace mpfx {

/**
 * @brief Status flags for rounded operations.
 * 
 * This class manages IEEE 754 status flags as bitfields within a single uint32_t.
 * 
 * Flag meanings:
 * 
 * - `invalid` - An operation had no usefully definable result.
 * 
 * - `div_by_zero` - An operation produced an exact infinite result for finite operands.
 * 
 * - `overflow` - An operation would have produced a value that was larger in magnitude
 * than the largest representable finite value were the exponent range unbounded.
 * 
 * - `tiny_before_rounding` - An operation would have produced a value that was smaller
 * in magnitude than the smallest representable normalized value without rounding.
 * 
 * - `tiny_after_rounding` - An operation would have produced a value that was smaller
 * in magnitude than the smallest representable normalized value were the exponent range unbounded.
 * 
 * - `underflow_before_rounding` - An operation was tiny (before rounding) and inexact.
 * 
 * - `underflow_after_rounding` - An operation was tiny (after rounding) and inexact.
 * 
 * - `inexact` - An operation produced a result that is different than the numerical result
 * if both precision and exponent range were unbounded.
 * 
 * - `carry` - An operation produced a result with a different (normalized) exponent than
 * the result if the precision were unbounded.
 */
class Flags {
private:
    uint32_t flags;

    static constexpr uint32_t INVALID_FLAG = 1u << 0;
    static constexpr uint32_t DIV_BY_ZERO_FLAG = 1u << 1;
    static constexpr uint32_t OVERFLOW_FLAG = 1u << 2;
    static constexpr uint32_t TINY_BEFORE_ROUNDING_FLAG = 1u << 3;
    static constexpr uint32_t TINY_AFTER_ROUNDING_FLAG = 1u << 4;
    static constexpr uint32_t UNDERFLOW_BEFORE_ROUNDING_FLAG = 1u << 5;
    static constexpr uint32_t UNDERFLOW_AFTER_ROUNDING_FLAG = 1u << 6;
    static constexpr uint32_t INEXACT_FLAG = 1u << 7;
    static constexpr uint32_t CARRY_FLAG = 1u << 8;

public:
    /// @brief Constructor initializes all flags to false.
    Flags() : flags(0) {}

    /// @brief Check if invalid flag is set.
    bool invalid() const { return flags & INVALID_FLAG; }
    /// @brief Set invalid flag.
    void set_invalid() { flags |= INVALID_FLAG; }

    /// @brief Check if division by zero flag is set.
    bool div_by_zero() const { return flags & DIV_BY_ZERO_FLAG; }
    /// @brief Set division by zero flag.
    void set_div_by_zero() { flags |= DIV_BY_ZERO_FLAG; }

    /// @brief Check if overflow flag is set.
    bool overflow() const { return flags & OVERFLOW_FLAG; }
    /// @brief Set overflow flag.
    void set_overflow() { flags |= OVERFLOW_FLAG; }

    /// @brief Check if tiny before rounding flag is set.
    bool tiny_before_rounding() const { return flags & TINY_BEFORE_ROUNDING_FLAG; }
    /// @brief Set tiny before rounding flag.
    void set_tiny_before_rounding() { flags |= TINY_BEFORE_ROUNDING_FLAG; }

    /// @brief Check if tiny after rounding flag is set.
    bool tiny_after_rounding() const { return flags & TINY_AFTER_ROUNDING_FLAG; }
    /// @brief Set tiny after rounding flag.
    void set_tiny_after_rounding() { flags |= TINY_AFTER_ROUNDING_FLAG; }

    /// @brief Check if underflow before rounding flag is set.
    bool underflow_before_rounding() const { return flags & UNDERFLOW_BEFORE_ROUNDING_FLAG; }
    /// @brief Set underflow before rounding flag.
    void set_underflow_before_rounding() { flags |= UNDERFLOW_BEFORE_ROUNDING_FLAG; }

    /// @brief Check if underflow after rounding flag is set.
    bool underflow_after_rounding() const { return flags & UNDERFLOW_AFTER_ROUNDING_FLAG; }
    /// @brief Set underflow after rounding flag.
    void set_underflow_after_rounding() { flags |= UNDERFLOW_AFTER_ROUNDING_FLAG; }

    /// @brief Check if inexact flag is set.
    bool inexact() const { return flags & INEXACT_FLAG; }
    /// @brief Set inexact flag.
    void set_inexact() { flags |= INEXACT_FLAG; }

    /// @brief Check if carry flag is set.
    bool carry() const { return flags & CARRY_FLAG; }
    /// @brief Set carry flag.
    void set_carry() { flags |= CARRY_FLAG; }

    /// @brief Reset all status flags to false.
    void reset() { flags = 0; }
};

/// @brief Global flags instance.
extern Flags flags;

} // end namespace mpfx
