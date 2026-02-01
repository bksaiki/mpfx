#pragma once

#include "context.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief MPFR-style floating-point rounding context.
///
/// This context represents floating-point values with an arbitrary precision,
/// no exponent bounds, and a specified rounding mode.
class MPContext : public Context {
public:
    MPContext(prec_t prec, RM rm) : Context(prec, std::nullopt, std::nullopt, rm) {}
};

} // namespace mpfx
