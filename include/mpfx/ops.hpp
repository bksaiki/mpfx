#pragma once

#include <cmath>

#include "context.hpp"
#include "engine_fp.hpp"
#include "engine_fpe.hpp"
#include "engine_fx.hpp"
#include "round.hpp"

namespace mpfx {

/// @brief Engine types for arithmetic operations
enum class EngineType {
    FP_RTO,   // Native floating-point using RTO emulation
    FP_EXACT, // Exact computation engine
    FIXED     // Fixed-point arithmetic engine
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
template<EngineType E = EngineType::FP_RTO>
double add(double x, double y, const Context& ctx) {
    if constexpr (E == EngineType::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::add(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::FP_EXACT) {
        // compute result using exact engine
        const double r = engine_fpe::add(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    }
}

/// @brief Computes `x - y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::FP_RTO>
double sub(double x, double y, const Context& ctx) {
    if constexpr (E == EngineType::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::sub(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::FP_EXACT) {
        // compute result using exact engine
        const double r = engine_fpe::sub(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    }
}

/// @brief Computes `x * y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::FP_RTO>
double mul(double x, double y, const Context& ctx) {
    const prec_t p = ctx.round_prec();
    if constexpr (E == EngineType::FP_RTO) {
        // compute result using RTO engine
        const double r = engine_fp::mul(x, y, p);
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::FP_EXACT) {
        // compute result using exact engine
        const double r = engine_fpe::mul(x, y, p);
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::FIXED) {
        // compute result using fixed-point arithmetic engine
        if (std::isfinite(x) && std::isfinite(y)) {
            // we can use fixed-point arithmetic
            auto [m, exp] = engine_fx::mul(x, y, p);
            // round the fixed-point result
            return ctx.round(m, exp);
        } else {
            // special value so use exact engine
            const double r = engine_fpe::mul(x, y, p);
            // use context to round
            return ctx.round(r);
        }
    }
}

/// @brief Computes `x / y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double div(double x, double y, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine_fp::div(x, y, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

/// @brief Computes `sqrt(x)` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double sqrt(double x, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine_fp::sqrt(x, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

/// @brief Computes `x * y + z` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double fma(double x, double y, double z, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine_fp::fma(x, y, z, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

} // end namespace mpfx
