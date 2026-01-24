#pragma once

#include <cmath>

#include "context.hpp"
#include "engine.hpp"
#include "round.hpp"

namespace mpfx {

/// @brief Engine types for arithmetic operations
enum class EngineType {
    RTO,    // Round-to-odd engine  
    EXACT,  // Exact computation engine
    FIXED   // Fixed-point arithmetic engine
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
template<EngineType E = EngineType::RTO>
double add(double x, double y, const Context& ctx) {
    if constexpr (E == EngineType::RTO) {
        // compute result using RTO engine
        const double r = engine::add(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::EXACT) {
        // compute result using exact engine
        const double r = engine::add_exact(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    }
}

/// @brief Computes `x - y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::RTO>
double sub(double x, double y, const Context& ctx) {
    if constexpr (E == EngineType::RTO) {
        // compute result using RTO engine
        const double r = engine::sub(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::EXACT) {
        // compute result using exact engine
        const double r = engine::sub_exact(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    }
}

/// @brief Computes `x * y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::RTO>
double mul(double x, double y, const Context& ctx) {
    const prec_t p = ctx.round_prec();
    if constexpr (E == EngineType::RTO) {
        // compute result using RTO engine
        const double r = engine::mul(x, y, p);
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::EXACT) {
        // compute result using exact engine
        const double r = engine::mul_exact(x, y, p);
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::FIXED) {
        // compute result using fixed-point arithmetic engine
        if (std::isfinite(x) && std::isfinite(y)) {
            // we can use fixed-point arithmetic
            auto [m, exp] = engine::mul_fixed(x, y, p);
            // round the fixed-point result
            return ctx.round(m, exp);
        } else {
            // special value so use exact engine
            const double r = engine::mul_exact(x, y, p);
            // use context to round
            return ctx.round(r);
        }
    }
}

/// @brief Computes `x / y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double div(double x, double y, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::div(x, y, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

/// @brief Computes `sqrt(x)` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double sqrt(double x, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::sqrt(x, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

/// @brief Computes `x * y + z` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
inline double fma(double x, double y, double z, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::fma(x, y, z, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

} // end namespace mpfx
