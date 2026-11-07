#pragma once

#include <cmath>
#include <concepts>
#include <type_traits>

#include "context.hpp"
#include "engine_eft.hpp"
#include "engine_ff.hpp"
#include "flags.hpp"
#include "engine_fp.hpp"
#include "engine_fx.hpp"
#include "engine_sf.hpp"
#include "round.hpp"

namespace mpfx {

namespace detail {

/// @brief Sign of an exact zero sum (IEEE 754-2019 §6.3).
/// An exact zero sum is +0 under every rounding direction except RTN, where it is -0.
/// Terms that are all zeros of one sign keep that sign. The engines already give +0
/// (or the terms' shared sign), so only RTN needs a fix.
/// @param r unrounded result
/// @param rm rounding mode of the target format
/// @param a first term
/// @param rest remaining terms
template<std::floating_point T, std::same_as<T>... Ts>
inline T exact_zero_sum(T r, RM rm, T a, Ts... rest) {
    if (r != 0 || rm != RM::RTN) {
        return r;
    }
    const bool one_sign = (a == 0) && (... && (rest == 0))
        && (... && (std::signbit(rest) == std::signbit(a)));
    return one_sign ? r : -static_cast<T>(0);
}

/// @brief Sign of an exact zero `x * y + z`, by the rule for sums.
/// The product is a term of sign `signbit(x) != signbit(y)`, and is zero only if `x` or `y` is,
/// so no multiplication is needed.
template<std::floating_point T>
inline T exact_zero_fma(T r, RM rm, T x, T y, T z) {
    if (r != 0 || rm != RM::RTN) {
        return r;
    }
    const bool one_sign = (x == 0 || y == 0) && z == 0
        && (std::signbit(x) != std::signbit(y)) == std::signbit(z);
    return one_sign ? r : -static_cast<T>(0);
}

} // end namespace detail

/// @brief Engine types for arithmetic operations
enum class Engine {
    FP_RTO,    // Native floating-point using RTO emulation
    FP_EXACT,  // Exact computation engine
    FIXED,     // Fixed-point arithmetic engine
    SOFTFLOAT, // SoftFloat engine
    FFLOAT,    // FloppyFloat engine
    EFT        // Error-free transformation engine
};

/// @brief Whether arithmetic over type `T` is supported by engine `E`.
///
/// Only `double` is supported by every engine. The FP_EXACT and fixed-point
/// engines are `double`-only here, so non-`double` types are limited to the
/// engines that support multiple precisions: EFT, FP_RTO, SoftFloat, and
/// FloppyFloat.
template <Engine E, std::floating_point T>
inline constexpr bool engine_supports_type =
    std::is_same_v<T, double>
    || E == Engine::EFT
    || E == Engine::FP_RTO
    || E == Engine::SOFTFLOAT
    || E == Engine::FFLOAT;

/// @brief Rounds `x` according to the given context.
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x a number to round
/// @param ctx rounding context
/// @return the rounded number
template <flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T round(T x, const Context& ctx) {
    return ctx.round<FlagMask>(x);
}

/// @brief Computes `-x` using the given context.
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x a number to negate
/// @param ctx rounding context
/// @return the negated number
template <flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T neg(T x, const Context& ctx) {
    // negate exactly
    x = -x;

    // use context to round
    return ctx.round<FlagMask>(x);
}

/// @brief Computes `|x|` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x a number
/// @param ctx rounding context
/// @return the absolute value
template <flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T abs(T x, const Context& ctx) {
    // take absolute value exactly
    x = std::abs(x);

    // use context to round
    return ctx.round<FlagMask>(x);
}

/// @brief Computes `x + y` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x first operand
/// @param y second operand
/// @param ctx rounding context
/// @return the sum
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T add(T x, T y, const Context& ctx) {
    static_assert(engine_supports_type<E, T>,
                  "non-double arithmetic requires a multi-precision engine (EFT, FP_RTO, SoftFloat, or FloppyFloat)");
    T result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const T r = engine_fp::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y));
    } else if constexpr (E == Engine::FP_EXACT) {
        // compute result using exact engine
        const T r = x + y;
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y));
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const T r = engine_sf::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y));
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const T r = engine_ff::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y));
    } else if constexpr (E == Engine::EFT) {
        // compute result using Error-Free Transformation engine
        const T r = engine_eft::add(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y));
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            if (std::isinf(x) && std::isinf(y) && (std::signbit(x) != std::signbit(y))) {
                // invalid operation: inf + -inf
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x - y` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x first operand
/// @param y second operand
/// @param ctx rounding context
/// @return the difference
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T sub(T x, T y, const Context& ctx) {
    static_assert(engine_supports_type<E, T>,
                  "non-double arithmetic requires a multi-precision engine (EFT, FP_RTO, SoftFloat, or FloppyFloat)");
    T result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const T r = engine_fp::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, -y));
    } else if constexpr (E == Engine::FP_EXACT) {
        // compute result using exact engine
        const T r = x - y;
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, -y));
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const T r = engine_sf::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, -y));
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const T r = engine_ff::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, -y));
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::sub(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, -y));
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            if (std::isinf(x) && std::isinf(y) && (std::signbit(x) == std::signbit(y))) {
                // invalid operation: inf - inf
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x * y` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x first operand
/// @param y second operand
/// @param ctx rounding context
/// @return the product
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T mul(T x, T y, const Context& ctx) {
    static_assert(engine_supports_type<E, T>,
                  "non-double arithmetic requires a multi-precision engine (EFT, FP_RTO, SoftFloat, or FloppyFloat)");
    const prec_t p = ctx.round_prec();
    T result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const T r = engine_fp::mul(x, y, p);
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::FP_EXACT) {
        // compute result using exact engine
        const T r = x * y;
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::FIXED) {
        // compute result using fixed-point arithmetic engine
        if (std::isfinite(x) && std::isfinite(y)) {
            // we can use fixed-point arithmetic
            auto [m, exp] = engine_fx::mul(x, y, p);
            // round the fixed-point result
            result = ctx.round<FlagMask>(m, exp);
        } else {
            // special value so use exact engine
            const T r = x * y;
            // use context to round
            result = ctx.round<FlagMask>(r);
        }
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const T r = engine_sf::mul(x, y, p);
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const T r = engine_ff::mul(x, y, p);
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::mul(x, y, p);
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            if ((x == 0.0 && std::isinf(y)) || (std::isinf(x) && y == 0.0)) {
                // invalid operation: 0 * inf
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x / y` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x dividend
/// @param y divisor
/// @param ctx rounding context
/// @return the quotient
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T div(T x, T y, const Context& ctx) {
    static_assert(engine_supports_type<E, T>,
                  "non-double arithmetic requires a multi-precision engine (EFT, FP_RTO, SoftFloat, or FloppyFloat)");
    T result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const T r = engine_fp::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const T r = engine_sf::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const T r = engine_ff::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::div(x, y, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            if ((x == 0.0 && y == 0.0) || (std::isinf(x) && std::isinf(y))) {
                // invalid operation: 0/0 or inf/inf
                flags.set_invalid();
            }
        }
    }

    if constexpr (FlagMask & Flags::DIV_BY_ZERO_FLAG) {
        if (std::isfinite(x) && x != 0.0 && y == 0.0) {
            // division by zero: finite non-zero / 0
            flags.set_div_by_zero();
        }
    }

    return result;
}

/// @brief Computes `sqrt(x)` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x the radicand
/// @param ctx rounding context
/// @return the square root
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T sqrt(T x, const Context& ctx) {
    static_assert(engine_supports_type<E, T>,
                  "non-double arithmetic requires a multi-precision engine (EFT, FP_RTO, SoftFloat, or FloppyFloat)");
    T result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const T r = engine_fp::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const T r = engine_sf::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const T r = engine_ff::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::sqrt(x, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(r);
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            if (x < 0.0) {
                // invalid operation: sqrt of a negative number, -inf included
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x * y + z` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x first multiplicand
/// @param y second multiplicand
/// @param z addend
/// @param ctx rounding context
/// @return the fused multiply-add result
template<Engine E = Engine::FP_RTO, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T fma(T x, T y, T z, const Context& ctx) {
    static_assert(engine_supports_type<E, T>,
                  "non-double arithmetic requires a multi-precision engine (EFT, FP_RTO, SoftFloat, or FloppyFloat)");
    T result;

    if constexpr (E == Engine::FP_RTO) {
        // compute result using RTO engine
        const T r = engine_fp::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_fma(r, ctx.rm(), x, y, z));
    } else if constexpr (E == Engine::SOFTFLOAT) {
        // compute result using SoftFloat engine
        const T r = engine_sf::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_fma(r, ctx.rm(), x, y, z));
    } else if constexpr (E == Engine::FFLOAT) {
        // compute result using FloppyFloat engine
        const T r = engine_ff::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_fma(r, ctx.rm(), x, y, z));
    } else if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::fma(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_fma(r, ctx.rm(), x, y, z));
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
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

/// @brief Computes `x + y + z` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x first operand
/// @param y second operand
/// @param z third operand
/// @param ctx rounding context
/// @return the sum
template<Engine E = Engine::EFT, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T add3(T x, T y, T z, const Context& ctx) {
    // only the EFT engine implements 3-operand summation (see the `else` below)
    T result;
    if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::add3(x, y, z, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y, z));
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            // check for invalid operation: inf + -inf = NaN
            const bool x_inf = std::isinf(x);
            const bool y_inf = std::isinf(y);
            const bool z_inf = std::isinf(z);
            const bool x_sign = std::signbit(x);
            const bool y_sign = std::signbit(y);
            const bool z_sign = std::signbit(z);
            if ((x_inf && y_inf && (x_sign != y_sign)) ||
                (x_inf && z_inf && (x_sign != z_sign)) ||
                (y_inf && z_inf && (y_sign != z_sign))) {
                flags.set_invalid();
            }
        }
    }

    return result;
}

/// @brief Computes `x + y + z + w` using the given context.
/// Requires `ctx.round_prec()` to fit `T`'s precision (53 for `double`, 24 for `float`).
/// @tparam E engine to use for computation
/// @tparam FlagMask mask to indicate the status flags to check during rounding.
/// @param x first operand
/// @param y second operand
/// @param z third operand
/// @param w fourth operand
/// @param ctx rounding context
/// @return the sum
template<Engine E = Engine::EFT, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T = double>
T add4(T x, T y, T z, T w, const Context& ctx) {
    // only the EFT engine implements 4-operand summation (see the `else` below)
    T result;
    if constexpr (E == Engine::EFT) {
        // compute result using error-free transformations
        const T r = engine_eft::add4(x, y, z, w, ctx.round_prec());
        // use context to round
        result = ctx.round<FlagMask>(detail::exact_zero_sum(r, ctx.rm(), x, y, z, w));
    } else {
        MPFX_STATIC_ASSERT(false, "Unsupported engine");
    }

    // check for special values to raise status flags
    if constexpr (FlagMask & Flags::INVALID_FLAG) {
        if (std::isnan(result)) [[unlikely]] {
            // check for invalid operation: inf + -inf = NaN
            const bool x_inf = std::isinf(x);
            const bool y_inf = std::isinf(y);
            const bool z_inf = std::isinf(z);
            const bool w_inf = std::isinf(w);
            const bool x_sign = std::signbit(x);
            const bool y_sign = std::signbit(y);
            const bool z_sign = std::signbit(z);
            const bool w_sign = std::signbit(w);
            if ((x_inf && y_inf && (x_sign != y_sign)) ||
                (x_inf && z_inf && (x_sign != z_sign)) ||
                (x_inf && w_inf && (x_sign != w_sign)) ||
                (y_inf && z_inf && (y_sign != z_sign)) ||
                (y_inf && w_inf && (y_sign != w_sign)) ||
                (z_inf && w_inf && (z_sign != w_sign))) {
                flags.set_invalid();
            }
        }
    }

    return result;
}

} // end namespace mpfx
