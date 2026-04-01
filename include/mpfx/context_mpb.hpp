#pragma once

#include <cmath>

#include "context.hpp"
#include "params.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief MPFR-style floating-point rounding context with minimum exponent and overflow.
///
/// This context represents floating-point values with an arbitrary precision,
/// a minimum exponent bound, a specified rounding mode, and a maximum representable
/// magnitude. When a value exceeds this maximum magnitude, it is treated as an overflow.
class MPBContext : public Context {
public:
    constexpr MPBContext(prec_t prec, exp_t emin, double maxval, RM rm)
        : Context(prec, emin - static_cast<exp_t>(prec), maxval, rm), emin_(emin) {}

    /// @brief Gets the minimum exponent of this context.
    constexpr exp_t emin() const {
        return emin_;
    }

    /// @brief Gets the maximum exponent of this context.
    inline exp_t emax() const {
        using FP = float_params<double>::params;
        const double maxval = *maxval_;
        return maxval == 0.0 ? FP::EXPMIN : static_cast<exp_t>(std::ilogb(maxval));
    }

private:

    /// @brief The minimum (normalized) exponent.
    exp_t emin_;
};

} // namespace mpfx
