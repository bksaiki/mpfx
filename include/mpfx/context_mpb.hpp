#pragma once

#include <cmath>

#include "context_mps.hpp"
#include "types.hpp"

namespace mpfx {

namespace {

inline bool overflow_to_infinity(RM rm, bool s, bool maxval_odd) {
    const auto dir = get_direction(rm, s);
    switch (dir)
    {
    case RoundingDirection::TO_ZERO:
        // always round towards zero
        return false;
    case RoundingDirection::AWAY_ZERO:
        // always round away from zero
        return true;
    case RoundingDirection::TO_EVEN:
        // round to infinity if maxval is odd
        return maxval_odd;
    case RoundingDirection::TO_ODD:
        // round to infinity if maxval is even
        return !maxval_odd;
    default:
        MPFX_UNREACHABLE("invalid rounding direction");
    }
}

inline double round_overflow(double x, RM rm, double maxval, bool maxval_odd) {
    // Handle special cases first
    if (!std::isfinite(x)) {
        return x; // NaN or exact infinity
    }

    // Check for overflow
    if (std::abs(x) > maxval) {
        // set the overflow flag and inexact flag
        overflow_flag = true;
        inexact_flag = true;

        // should we round to infinity?
        const bool s = std::signbit(x);
        if (overflow_to_infinity(rm, s, maxval_odd)) {
            // round to infinity
            static constexpr double POS_INF = std::numeric_limits<double>::infinity();
            return std::copysign(POS_INF, x);
        } else {
            // round to maxval
            return std::copysign(maxval, x);
        }
    }

    return x;
}

} // anonymous namespace

/// @brief MPFR-style floating-point rounding context with minimum exponent and overflow.
///
/// This context represents floating-point values with an arbitrary precision,
/// a minimum exponent bound, a specified rounding mode, and a maximum representable
/// magnitude. When a value exceeds this maximum magnitude, it is treated as an overflow.
class MPBContext : public Context {
private:
    /// @brief Underlying MPS context for subnormalization handling.
    MPSContext mps_ctx_;
    /// @brief The overflow value (maximum representable magnitude).
    double maxval_;
    /// @brief Is the maximum value odd?
    bool maxval_odd_;

public:

    inline MPBContext(prec_t prec, exp_t emin, RM rm, double maxval)
        : mps_ctx_(prec, emin, rm), maxval_(maxval) {
        using FP = ieee754_consts<11, 64>; // IEEE 754 double precision

        // check that the maximum value is valid
        MPFX_ASSERT(!std::signbit(maxval), "maxval must be non-negative");
        MPFX_ASSERT(std::isfinite(maxval), "maxval must be finite");
        MPFX_ASSERT(maxval == mps_ctx_.round(maxval), "maxval must be exactly representable in this context");

        // check if the maximum value is odd
        const uint64_t bits = std::bit_cast<uint64_t>(maxval);
        const int pth_bit_pos = static_cast<int>(FP::M) - static_cast<int>(prec) + 1;
        maxval_odd_ = (pth_bit_pos >= 0) && ((bits >> pth_bit_pos) & 1);
    }

    /// @brief Gets the maximum precision of this context.
    inline prec_t prec() const {
        return mps_ctx_.prec();
    }

    /// @brief Gets the minimum exponent of this context.
    inline exp_t emin() const {
        return mps_ctx_.emin();
    }

    /// @brief Gets the maximum exponent of this context.
    inline exp_t emax() const {
        return maxval_ == 0.0 ? 0 : std::ilogb(maxval_);
    }

    /// @brief Gets the rounding mode of this context.
    inline RM rm() const {
        return mps_ctx_.rm();
    }

    /// @brief Gets the precomputed subnormalization parameter (n = emin - p).
    inline exp_t n() const {
        return mps_ctx_.n();
    }

    /// @brief This context without overflow handling.
    inline const MPSContext& mps_context() const {
        return mps_ctx_;
    }

    /// @brief Gets the overflow value (maximum representable magnitude).
    inline double maxval() const {
        return maxval_;
    }

    /// @brief Returns true if maxval is odd (pth bit of mantissa is set).
    inline bool maxval_odd() const {
        return maxval_odd_;
    }

    inline prec_t round_prec() const override {
        return mps_ctx_.round_prec();
    }

    inline double round(double x) const override {
        // round without overflow handling
        x = mps_ctx_.round(x);

        // apply overflow handling
        return round_overflow(x, mps_ctx_.rm(), maxval_, maxval_odd_);
    }

    inline double round(int64_t m, exp_t exp) const override {
        // round without overflow handling
        double x = mps_ctx_.round(m, exp);

        // apply overflow handling
        return round_overflow(x, mps_ctx_.rm(), maxval_, maxval_odd_);
    }
};

} // namespace mpfx
