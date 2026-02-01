#pragma once

#include <cmath>

#include "context.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief MPFR-style floating-point rounding context with minimum exponent and overflow.
///
/// This context represents floating-point values with an arbitrary precision,
/// a minimum exponent bound, a specified rounding mode, and a maximum representable
/// magnitude. When a value exceeds this maximum magnitude, it is treated as an overflow.
class MPBContext : public Context {
private:
    /// @brief The minimum (normalized) exponent.
    exp_t emin_;

    /// @brief The maximum (normalized) exponent.
    exp_t emax_;

public:
    MPBContext(prec_t prec, exp_t emin, RM rm, double maxval);

    /// @brief Gets the minimum exponent of this context.
    inline exp_t emin() const {
        return emin_;
    }

    /// @brief Gets the maximum exponent of this context.
    inline exp_t emax() const {
        return emax_;
    }
};

} // namespace mpfx
