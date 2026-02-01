#pragma once

#include <cmath>
#include <optional>

#include "flags.hpp"
#include "params.hpp"
#include "round.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief Rounding context
///
/// A rounding context encapsulates a rounding operation from real numbers
/// to a number representation. In particular, this rounding context
/// specifies number representations with a finite precision, an optional
/// first unrepresented digit, and an optional largest representable value.
class Context {
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
    /// @brief Is the maximum value odd? (used for overflow tie-breaking)
    bool maxval_is_odd_;

private:
    /// @brief Helper function to determine if overflow rounds to infinity.
    static inline bool overflow_to_infinity(RM rm, bool s, bool maxval_odd) {
        const auto dir = get_direction(rm, s);
        switch (dir)
        {
        case RoundingDirection::TO_ZERO:
            return false;
        case RoundingDirection::AWAY_ZERO:
            return true;
        case RoundingDirection::TO_EVEN:
            return maxval_odd;
        case RoundingDirection::TO_ODD:
            return !maxval_odd;
        default:
            MPFX_UNREACHABLE("invalid rounding direction");
        }
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

        const double maxval = maxval_.value();
        if (std::abs(x) > maxval) {
            if constexpr (FlagMask & Flags::OVERFLOW_FLAG) {
                flags.set_overflow();
            }
            if constexpr (FlagMask & Flags::INEXACT_FLAG) {
                flags.set_inexact();
            }

            const bool s = std::signbit(x);
            if (overflow_to_infinity(rm_, s, maxval_is_odd_)) {
                static constexpr double POS_INF = std::numeric_limits<double>::infinity();
                return std::copysign(POS_INF, x);
            } else {
                return std::copysign(maxval, x);
            }
        }

        return x;
    }

public:

    /// @brief Constructs a Context instance with given parameters.
    /// @param p precision
    /// @param n first unrepresented digit
    /// @param maxval maximum representable magnitude
    /// @param rm rounding mode
    Context(prec_t p, const std::optional<exp_t>& n, const std::optional<double>& maxval, RM rm);

    /// @brief Gets the precision of this context.
    inline prec_t prec() const {
        return p_;
    }

    /// @brief Gets the rounding mode of this context.
    inline RM rm() const {
        return rm_;
    }

    /// @brief Gets the first unrepresented digit (subnormalization parameter).
    inline std::optional<exp_t> n() const {
        return n_;
    }

    /// @brief Gets the maximum representable magnitude.
    inline std::optional<double> maxval() const {
        return maxval_;
    }

    /// @brief Returns true if maxval is odd (pth bit of mantissa is set).
    inline bool maxval_is_odd() const {
        return maxval_is_odd_;
    }

    /// @brief Minimum precision using round-to-odd required for
    /// safe rerounding under this rounding context.
    inline prec_t round_prec() const {
        return p_ + 2;
    }

    /// @brief Rounds `x` according to this rounding context.
    /// @tparam Mask to indicate the status flags to check during rounding.
    /// @param x a number to round
    /// @return the rounded number
    template <flag_mask_t FlagMask = Flags::ALL_FLAGS>
    double round(double x) const {
        x = mpfx::round<FlagMask>(x, p_, n_, rm_);
        return round_overflow<FlagMask>(x);
    }

    /// @brief Rounds `m * 2^exp` according to this rounding context.
    /// @tparam Mask to indicate the status flags to check during rounding.
    /// @param m integer significand
    /// @param exp base-2 exponent
    /// @return the rounded number
    template <flag_mask_t FlagMask = Flags::ALL_FLAGS>
    double round(int64_t m, exp_t exp) const {
        double x = mpfx::round<FlagMask>(m, exp, p_, n_, rm_);
        return round_overflow<FlagMask>(x);
    }
};

} // namespace mpfx
