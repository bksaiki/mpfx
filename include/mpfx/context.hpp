#pragma once

#include <cmath>
#include <limits>
#include <optional>

#include "flags.hpp"
#include "params.hpp"
#include "round.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief Behavior when a value exceeds the maximum representable magnitude.
enum class OverflowMode : uint8_t {
    /// @brief Overflow rounds to infinity or `maxval` depending on the
    /// rounding direction (the usual IEEE 754 behavior).
    OVERFLOW,
    /// @brief Overflow always saturates to `±maxval`, never producing
    /// infinity. Used by formats without infinities.
    SATURATE,
};

/// @brief Rounding context
///
/// A rounding context encapsulates a rounding operation from real numbers
/// to a number representation. In particular, this rounding context
/// specifies number representations with a finite precision, an optional
/// first unrepresented digit, and an optional largest representable value.
class Context {
public:

    /// @brief Constructs a Context instance with given parameters.
    /// @param p precision
    /// @param n first unrepresented digit
    /// @param maxval maximum representable magnitude
    /// @param rm rounding mode
    /// @param overflow behavior when a value exceeds `maxval`
    constexpr Context(
        prec_t p,
        const std::optional<exp_t>& n,
        const std::optional<double>& maxval,
        RM rm,
        OverflowMode overflow = OverflowMode::OVERFLOW)
        : p_(p), n_(n), maxval_(maxval), rm_(rm), overflow_(overflow) {}

    /// @brief Gets the precision of this context.
    constexpr prec_t prec() const {
        return p_;
    }

    /// @brief Gets the rounding mode of this context.
    constexpr RM rm() const {
        return rm_;
    }

    /// @brief Gets the overflow behavior of this context.
    constexpr OverflowMode overflow() const {
        return overflow_;
    }

    /// @brief Gets the first unrepresented digit (subnormalization parameter).
    constexpr std::optional<exp_t> n() const {
        return n_;
    }

    /// @brief Gets the maximum representable magnitude.
    constexpr std::optional<double> maxval() const {
        return maxval_;
    }

    /// @brief Minimum precision using round-to-odd required for
    /// safe rerounding under this rounding context.
    constexpr prec_t round_prec() const {
        return p_ + 2;
    }

    /// @brief Rounds `x` according to this rounding context.
    /// @tparam FlagMask mask to indicate the status flags to check during rounding.
    /// @param x a number to round
    /// @return the rounded number
    template <flag_mask_t FlagMask = Flags::ALL_FLAGS>
    double round(double x) const {
        x = mpfx::round<FlagMask>(x, p_, n_, rm_);
        return round_overflow<FlagMask>(x);
    }

    /// @brief Rounds `m * 2^exp` according to this rounding context.
    /// @tparam FlagMask mask to indicate the status flags to check during rounding.
    /// @param m integer significand
    /// @param exp base-2 exponent
    /// @return the rounded number
    template <flag_mask_t FlagMask = Flags::ALL_FLAGS, signed_integral T>
    double round(T m, exp_t exp) const {
        double x = mpfx::round<FlagMask, T>(m, exp, p_, n_, rm_);
        return round_overflow<FlagMask>(x);
    }

protected:

    /// @brief Precision of this context.
    prec_t p_;
    /// @brief First unrepresented digit. All significant digits must be
    /// to the left of this position.
    std::optional<exp_t> n_;
    /// @brief Largest representable value (for overflow handling).
    std::optional<double> maxval_;
    /// @brief Rounding mode of this context.
    RM rm_;
    /// @brief Overflow behavior of this context.
    OverflowMode overflow_;

private:

    /// @brief Helper function to determine if overflow rounds to infinity.
    static inline bool overflow_to_infinity(RM rm, bool s) {
        const auto dir = get_direction(rm, s);
        return dir != RoundingDirection::TO_ZERO;
    }

    /// @brief Helper function to handle overflow.
    /// @tparam Mask to indicate the status flags to set during overflow handling.
    /// @param x a number to check for overflow
    template <flag_mask_t FlagMask>
    double round_overflow(double x) const {
        if (!maxval_.has_value()) {
            return x;
        }

        if (!std::isfinite(x)) {
            return x;
        }

        const double maxval = *maxval_;
        if (std::abs(x) > maxval) {
            if constexpr (FlagMask & Flags::OVERFLOW_FLAG) {
                flags.set_overflow();
            }
            if constexpr (FlagMask & Flags::INEXACT_FLAG) {
                flags.set_inexact();
            }

            const bool s = std::signbit(x);
            if (overflow_ != OverflowMode::SATURATE && overflow_to_infinity(rm_, s)) {
                static constexpr double POS_INF = std::numeric_limits<double>::infinity();
                return std::copysign(POS_INF, x);
            } else {
                return std::copysign(maxval, x);
            }
        }

        return x;
    }
};

} // namespace mpfx
