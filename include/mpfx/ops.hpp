#pragma once

#include <cmath>

#include "context.hpp"
#include "engine_eft.hpp"
#include "engine_ff.hpp"
#include "flags.hpp"
#include "engine_fp.hpp"
#include "engine_fpe.hpp"
#include "engine_fx.hpp"
#include "engine_sf.hpp"
#include "round.hpp"

namespace mpfx {

/// @brief Engine types for arithmetic operations
enum class Engine {
    FP_RTO,    // Native floating-point using RTO emulation
    FP_EXACT,  // Exact computation engine
    FIXED,     // Fixed-point arithmetic engine
    SOFTFLOAT, // SoftFloat engine
    FFLOAT,    // FloppyFloat engine
    EFT        // Error-free transformation engine
};

/// @brief Rounds `x` according to the given context.
inline double round(double x, const Context& ctx) {
    return ctx.round(x);
}

/// @brief Computes `-x` using the given context.
inline double neg(double x, const Context& ctx) {
    // negate exactly
    x = -x;

    // use context to round
    return ctx.round(x);
}

/// @brief Computes `|x|` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double abs(double x, const Context& ctx) {
    // take absolute value exactly
    x = std::abs(x);

    // use context to round
    return ctx.round(x);
}

/// @brief Computes `x + y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double add(double x, double y, const Context& ctx) {
    double result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const double r = engine_sf::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const double r = engine_ff::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using Error-Free Transformation engine
        const double r = engine_eft::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID) {
        if (std::isnan(result)) {
            if (std::isinf(x) && std::isinf(y) && (std::signbit(x) != std::signbit(y))) {
                // invalid operation: inf + -inf
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x - y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double sub(double x, double y, const Context& ctx) {
    double result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const double r = engine_sf::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const double r = engine_ff::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const double r = engine_eft::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID) {
        if (std::isnan(result)) {
            if (std::isinf(x) && std::isinf(y) && (std::signbit(x) == std::signbit(y))) {
                // invalid operation: inf - inf
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x * y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double mul(double x, double y, const Context& ctx) {
    const prec_t p = ctx.round_prec();
    double result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::mul(x, y, p);
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FP_EXACT) {
        // compute result using exact engine
        const double r = engine_fpe::mul(x, y, p);
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FIXED) {
        // compute result using fixed-point arithmetic engine
        if (std::isfinite(x) && std::isfinite(y)) {
            // we can use fixed-point arithmetic
            auto [m, exp] = engine_fx::mul(x, y, p);
            // round the fixed-point result
            result = ctx.round(m, exp);
        } else {
            // special value so use exact engine
            const double r = engine_fpe::mul(x, y, p);
            // use context to round
            result = ctx.round(r);
        }
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const double r = engine_sf::mul(x, y, p);
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const double r = engine_ff::mul(x, y, p);
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const double r = engine_eft::mul(x, y, p);
        // use context to round
        result = ctx.round(r);
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID) {
        if (std::isnan(result)) {
            if ((x == 0.0 && std::isinf(y)) || (std::isinf(x) && y == 0.0)) {
                // invalid operation: 0 * inf
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x / y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double div(double x, double y, const Context& ctx) {
    double result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const double r = engine_sf::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const double r = engine_ff::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const double r = engine_eft::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID) {
        if (std::isnan(result)) {
            if ((x == 0.0 && y == 0.0) || (std::isinf(x) && std::isinf(y))) {
                // invalid operation: 0/0 or inf/inf
                flags.set_invalid();
            }
        }
    }

    if constexpr (FlagMask & Flags::DIV_BY_ZERO) {
        if (std::isfinite(x) && x != 0.0 && y == 0.0) {
            // division by zero: finite non-zero / 0
            flags.set_div_by_zero();
        }
    }

    return result;
}

/// @brief Computes `sqrt(x)` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double sqrt(double x, const Context& ctx) {
    double result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const double r = engine_sf::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const double r = engine_ff::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const double r = engine_eft::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID) {
        if (std::isnan(result)) {
            if (x < 0.0 && std::isfinite(x)) {
                // invalid operation: sqrt of negative number
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x * y + z` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double fma(double x, double y, double z, const Context& ctx) {
    double result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const double r = engine_sf::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const double r = engine_ff::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const double r = engine_eft::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round(r);
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID) {
        if (std::isnan(result)) {
            const bool x_nan = std::isnan(x);
            const bool y_nan = std::isnan(y);
            const bool x_inf = std::isinf(x);
            const bool y_inf = std::isinf(y);

            // Check for invalid multiplication (0 * inf)
            if ((x == 0.0 && y_inf) || (x_inf && y == 0.0)) {
                flags.set_invalid();
            } else if ((x_inf && !y_nan) || (y_inf && !x_nan)) {
                // product is infinite
                const double p = x * y;

                // Check for invalid addition (inf + -inf)
                if (std::isinf(z) && (std::signbit(p) != std::signbit(z))) {
                    flags.set_invalid();
                }
            }
        }
    }

    return result;
}

} // end namespace mpfx
