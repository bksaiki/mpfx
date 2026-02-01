#pragma once

#include "context.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief MPFR-style floating-point rounding context with minimum exponent.
///
/// This context represents floating-point values with an arbitrary precision,
/// a minimum exponent bound, and a specified rounding mode. Subnormalization
/// is computed as n = emin - p.
class MPSContext : public Context {
private:
    /// @brief The minimum (normalized) exponent.
    exp_t emin_;

public:
    MPSContext(prec_t prec, exp_t emin, RM rm)
        : Context(prec, emin - static_cast<exp_t>(prec), std::nullopt, rm), emin_(emin) {}

    /// @brief Gets the minimum exponent of this context.
    inline exp_t emin() const {
        return emin_;
    }
};

} // namespace mpfx
