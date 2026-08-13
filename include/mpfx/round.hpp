#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>

#include "bit_float.hpp"
#include "convert.hpp"
#include "flags.hpp"
#include "params.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief Rounding modes for floating-point operations
/// 
/// When a real value is not representable in the target format,
/// rounding modes determine which floating-point value to choose.
enum class RoundingMode : uint8_t {
    RNE,               // Round to nearest, ties to even
    RNA,               // Round to nearest, ties away from zero
    RTP,               // Round toward +infinity (ceiling)
    RTN,               // Round toward -infinity (floor)
    RTZ,               // Round toward zero (truncation)
    RAZ,               // Round away from zero
    RTO,               // Round to odd
    RTE,               // Round to even
};

/// @brief Alias for `RoundingMode`.
using RM = RoundingMode;

/// @brief Rounding direction
///
/// Indicates which value to round relative to the original value.
/// A `RoundingMode` can be mapped to a boolean indicating whether
/// the rounding is a nearest rounding and a `RoundingDirection`.
///
enum class RoundingDirection : uint8_t {
    TO_ZERO,
    AWAY_ZERO,
    TO_EVEN,
    TO_ODD,
};

/// @brief Returns the rounding direction for a given rounding mode and sign.
/// For nearest rounding modes, the direction is for tie-breaking.
inline RoundingDirection get_direction(RoundingMode mode, bool sign) {
    // Lookup table approach for better performance
    // Index: mode * 2 + sign
    static constexpr RoundingDirection table[] = {
        // RNE (mode 0): sign doesn't matter
        RoundingDirection::TO_EVEN, RoundingDirection::TO_EVEN,
        // RNA (mode 1): sign doesn't matter
        RoundingDirection::AWAY_ZERO, RoundingDirection::AWAY_ZERO,
        // RTP (mode 2): depends on sign
        RoundingDirection::AWAY_ZERO, RoundingDirection::TO_ZERO,
        // RTN (mode 3): depends on sign
        RoundingDirection::TO_ZERO, RoundingDirection::AWAY_ZERO,
        // RTZ (mode 4): sign doesn't matter
        RoundingDirection::TO_ZERO, RoundingDirection::TO_ZERO,
        // RAZ (mode 5): sign doesn't matter
        RoundingDirection::AWAY_ZERO, RoundingDirection::AWAY_ZERO,
        // RTO (mode 6): sign doesn't matter
        RoundingDirection::TO_ODD, RoundingDirection::TO_ODD,
        // RTE (mode 7): sign doesn't matter
        RoundingDirection::TO_EVEN, RoundingDirection::TO_EVEN,
    };

    if (static_cast<size_t>(mode) > static_cast<size_t>(RoundingMode::RTE)) [[unlikely]] {
        MPFX_DEBUG_ASSERT(false, "get_direction: invalid rounding mode");
        return RoundingDirection::TO_ZERO; // default return to avoid warnings
    }

    const size_t idx = (static_cast<size_t>(mode) << 1) | static_cast<size_t>(sign);
    return table[idx];
}

namespace experimental {

/// @brief Should we increment to round?
/// @tparam RM the rounding mode
/// @tparam T the type of the significand
/// @param hi the high part of the split significand
/// @param n the split point
/// @param rs the low part of the significand represented in a round-sticky scheme
/// @return should we increment the significand?
template <RM rm, std::floating_point T>
inline bool round_increment_nearest(bit_float<T> hi, exp_t n, RoundRS rs) {
    // case split on rounding mode
    if constexpr (rm == RM::RNE) {
        if (rs == RoundRS::EXACT_HALFWAY) {
            // exactly halfway - increment if the LSB is odd
            return hi.bit(n + 1);
        } else {
            // increment only if strictly above halfway
            return rs == RoundRS::ABOVE_HALFWAY;
        }
    } else if constexpr (rm == RM::RNA) {
        // above or exactly at halfway - increment
        return static_cast<uint8_t>(rs) >= static_cast<uint8_t>(RoundRS::EXACT_HALFWAY);
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        return false;
    }
}

/// @brief Should we increment to round?
/// @tparam RM the rounding mode
/// @tparam T the type of the significand
/// @param hi the high part of the split significand
/// @param lo the low part of the split significand
/// @param n the split point
/// @return should we increment the significand?
template <RM rm, std::floating_point T>
inline bool round_increment_directed(bit_float<T> hi, exp_t n) {
    // case split on rounding mode
    if constexpr (rm == RM::RTP) {
        // round toward +infinity
        return !hi.s();
    } else if constexpr (rm == RM::RTN) {
        // round toward -infinity
        return hi.s();
    } else if constexpr (rm == RM::RTZ) {
        // round toward zero
        return false;
    } else if constexpr (rm == RM::RAZ) {
        // round away from zero
        return true;
    } else if constexpr (rm == RM::RTO) {
        // round to odd => increment if LSB is even
        return !hi.bit(n + 1);
    } else if constexpr (rm == RM::RTE) {
        // round to even => increment if LSB is odd
        return hi.bit(n + 1);
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        return false;
    }
}

/// @brief Finalizes the rounding procedure.
/// @tparam T the floating-point type
template <std::floating_point T>
inline bit_float<T> round_finalize(bit_float<T> hi, exp_t exp, bool increment) {
    if (increment) {
        // increment the high part by adding "1" relative to the split point `n`
        return hi.next_away_zero(exp);
    } else {
        // no increment, just return the high part
        return hi;
    }
}

/// @brief Checks for tininess after rounding
/// @tparam T the floating-point type
/// @param x the original value (assumed to be tiny before rounding)
/// @param result the rounded value
/// @param e the normalized exponent of `x`
/// @param emin the minimum normalized exponent
/// @param n the actual split point used for rounding
template <RM rm, std::floating_point T>
inline bool round_tiny_after(bit_float<T> x, exp_t e, exp_t emin, exp_t n) {
    // below the largest subnormal binade - definitely tiny after rounding
    if (e < emin - 1) {
        return true;
    }

    // in the largest subnormal binade - possibly tiny after rounding
    const bit_float<T> min_norm = bit_float<T>::make_pow2(emin, x.s());
    const bit_float<T> cutoff = min_norm.next_toward_zero(n);
    if (x.compare_mag(cutoff) <= 0) {
        // we will never round up to 2^emin - definitely tiny after rounding
        return true;
    }

    // halfway to the smallest normal - round again with an additional bit
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        // nearest rounding modes
        const auto [hi, rs] = x.split_rs(n - 1);
        return !round_increment_nearest<rm>(hi, n - 1, rs);
    } else {
        // directed rounding modes
        const auto [hi, sticky] = x.split_sticky(n - 1);
        // tiny if `x` is not representable and it would have rounded down
        // with an additional bit of precision.
        return !(sticky && round_increment_directed<rm>(hi, n - 1));
    }
}

/// @brief Optimized rounding of a `bit_float` type.
/// @tparam RM the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
template <RM rm, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
bit_float<T> round(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // which flags to check
    static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_BEFORE = FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_AFTER = FlagMask & Flags::UNDERFLOW_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;

    // fast path: special values (infinity, NaN)
    if (x.is_nar()) {
        return x;
    }

    // fast path: zero
    if (x.is_zero()) {
        // raise tiny flags
        if constexpr (CHECK_TINY_BEFORE) {
            flags.set_tiny_before_rounding();
        }
        if constexpr (CHECK_TINY_AFTER) {
            flags.set_tiny_after_rounding();
        }

        return x;
    }

    // compute the actual split point `n`
    const exp_t e = x.e();
    const exp_t n_min = e - static_cast<exp_t>(p);
    const exp_t n_act = n.has_value() ? std::max(n_min, *n) : n_min;
    const exp_t emin = n.has_value() ? *n + static_cast<exp_t>(p) : std::numeric_limits<exp_t>::min();

    // set tiny before rounding flag if requested
    bool tiny_before = e < emin;
    if constexpr (CHECK_TINY_BEFORE) {
        if (tiny_before) {
            flags.set_tiny_before_rounding();
        }
    }

    // case split on rounding mode
    bit_float<T> result;
    bool increment;
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        // nearest rounding modes - need to recover lower part for tie-breaking
        // split the `bit_float` at the actual split point
        const auto [hi, rs] = x.split_rs(n_act);

        // fast path: low is zero
        if (rs == RoundRS::EXACT) {
            // we are tiny after rounding if we were tiny before rounding
            if constexpr (CHECK_TINY_AFTER) {
                if (tiny_before) {
                    flags.set_tiny_after_rounding();
                }
            }

            return hi;
        }

        // should we increment?
        increment = round_increment_nearest<rm>(hi, n_act, rs);
        result = round_finalize(hi, n_act + 1, increment);
    } else {
        // directed rounding mode - only need to check if `low == 0`
        // split the `bit_float` at the actual split point
        const auto [hi, sticky] = x.split_sticky(n_act);

        // fast path: low is zero
        if (!sticky) {
            // we are tiny after rounding if we were tiny before rounding
            if constexpr (CHECK_TINY_AFTER) {
                if (tiny_before) {
                    flags.set_tiny_after_rounding();
                }
            }

            return hi;
        }

        // should we increment?
        increment = round_increment_directed<rm>(hi, n_act);
        result = round_finalize(hi, n_act + 1, increment);
    }

    // set inexact flag if requested
    if constexpr (CHECK_INEXACT) {
        flags.set_inexact();
    }

    if (tiny_before) {
        // set underflow before rounding flag if requested
        if constexpr (CHECK_UNDERFLOW_BEFORE) {
            flags.set_underflow_before_rounding();
        }

        // detect tininess after rounding
        if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
            // we can only be tiny after rounding if we were tiny before rounding
            bool tiny_after = round_tiny_after<rm>(x, e, emin, n_act);

            if (tiny_after) {
                // set tiny after rounding flag if requested
                if constexpr (CHECK_TINY_AFTER) {
                    flags.set_tiny_after_rounding();
                }

                // set underflow after rounding flag if requested
                if constexpr (CHECK_UNDERFLOW_AFTER) {
                    flags.set_underflow_after_rounding();
                }
            }
        }
    } else {
        if constexpr (CHECK_CARRY) {
            // we can only carry if we increment (any not tiny before rounding)
            if (increment) {
                // we carry when the result is a power of two
                if (result.mbits() == 0) {
                    // set carry flag if requested
                    flags.set_carry();
                }
            }
        }
    }

    return result;
}

/// @brief Optimized rounding of a `bit_float` type.
/// @tparam RM the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
template <flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
bit_float<T> round(bit_float<T> x, prec_t p, std::optional<exp_t> n, RM rm) {
    switch (rm) {
    case RM::RNE:
        return round<RM::RNE, FlagMask>(x, p, n);
    case RM::RNA:
        return round<RM::RNA, FlagMask>(x, p, n);
    case RM::RTP:
        return round<RM::RTP, FlagMask>(x, p, n);
    case RM::RTN:
        return round<RM::RTN, FlagMask>(x, p, n);
    case RM::RTZ:
        return round<RM::RTZ, FlagMask>(x, p, n);
    case RM::RAZ:
        return round<RM::RAZ, FlagMask>(x, p, n);
    case RM::RTO:
        return round<RM::RTO, FlagMask>(x, p, n);
    case RM::RTE:
        return round<RM::RTE, FlagMask>(x, p, n);
    default:
        MPFX_DEBUG_ASSERT(false, "round: invalid rounding mode");
        return x; // default return to avoid warnings
    }
}

///
/// Scale-and-truncate rounding.
///
/// An alternative formulation of `round` that works in the floating-point
/// domain rather than the integer domain. Setting `exp = n + 1` and scaling `x`
/// by `2^-exp` puts the split point exactly at the binary point: the
/// representable digits of `x` become the integer part of the scaled value and
/// the unrepresentable digits become its fraction. Rounding is then a
/// round-to-integral operation, and the residual `y - trunc(y)` - which is
/// exact, with magnitude in `(0, 1)` and the sign of `x` - summarizes every
/// lost digit.
///

/// @brief Computes `x * 2^k` by adding `k` to the exponent field.
///
/// Assumes `x` is normal and that the result is normal or exactly `+/-Inf`.
/// Under those assumptions the mantissa is untouched and the sign bit is never
/// disturbed, so the scaling is exact. Both assumptions hold throughout
/// `round_scaled` whenever its input is normal: the scaled value `x * 2^-exp`
/// lands in `[1, 2^p)`, and rounding cannot change the normalized exponent
/// except by a carry to `e + 1`. In the carry-to-overflow case the significand
/// is a power of two, so the exponent field saturates onto `Inf` exactly.
template <std::floating_point T>
inline bit_float<T> scale_bits(bit_float<T> x, exp_t k) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    MPFX_DEBUG_ASSERT(!x.is_nar(), "cannot scale NaN or Inf");
    MPFX_DEBUG_ASSERT(x.ebits() != 0, "cannot scale a subnormal by exponent arithmetic");

    // `k` is cast before shifting so that negative `k` wraps in a
    // well-defined way; the wrapped value is the two's complement of `|k| * 2^M`
    const uint_t d = static_cast<uint_t>(k) << params_t::M;
    return bit_float<T>(static_cast<uint_t>(x.to_bits() + d));
}

/// @brief Computes `x * 2^k`, exact whenever the true product is representable.
///
/// The general fallback for `scale_bits`, needed when `x` is subnormal (its
/// exponent field is zero, so adding to it would fabricate an implicit bit) or
/// when the result is subnormal. Assumes `x` is finite.
///
/// A single multiply does not suffice: `exp` reaches down to `EXPMIN + 1`, and
/// the corresponding scale factor `2^-exp` is not representable. Two factors are
/// used instead, which supports `2 * EMIN <= k <= 2 * EMAX` - wider than every
/// scale rounding can ask for, as the static assertion below checks. Note this
/// is narrower than the range over which the product can be representable, so
/// this is not a general-purpose `ldexp`.
///
/// Both multiplies are exact. `k2` always carries the sign of the part of `k`
/// that was clamped away, so the intermediate holds the same significand as the
/// final result with its magnitude on the safe side of it: if the final result
/// is representable the intermediate is too, and neither over- nor underflows.
template <std::floating_point T>
inline T scale_mul(T x, exp_t k) {
    using params_t = typename float_params<T>::params;
    using uint_t = typename float_params<T>::uint_t;

    // rounding scales by `2^-exp` then by `2^exp`, for `exp` in
    // `[EXPMIN + 1, EMAX]`, so `k` never leaves `[EXPMIN + 1, -EXPMIN - 1]`
    MPFX_STATIC_ASSERT(2 * params_t::EMIN <= params_t::EXPMIN + 1,
                       "scale_mul cannot reach every scale rounding needs");
    MPFX_STATIC_ASSERT(2 * params_t::EMAX >= -params_t::EXPMIN - 1,
                       "scale_mul cannot reach every scale rounding needs");
    MPFX_DEBUG_ASSERT(k >= 2 * params_t::EMIN, "scale factor is too small");
    MPFX_DEBUG_ASSERT(k <= 2 * params_t::EMAX, "scale factor is too large");

    const exp_t k1 = std::clamp(k, params_t::EMIN, params_t::EMAX);
    const exp_t k2 = k - k1; // also within `[EMIN, EMAX]`
    const T c1 = std::bit_cast<T>(static_cast<uint_t>(k1 + params_t::BIAS) << params_t::M);
    const T c2 = std::bit_cast<T>(static_cast<uint_t>(k2 + params_t::BIAS) << params_t::M);
    return x * c1 * c2;
}

/// @brief Rounds `x` when every digit is unrepresentable, i.e. `n >= e`.
/// @tparam rm the rounding mode
/// @param x the value to round, neither zero nor NaN/Inf
/// @param e the normalized exponent of `x`
/// @param n the actual split point, at or above `e`
///
/// The scaled formulation cannot be used here: `x * 2^-(n+1)` has magnitude
/// below one, at an exponent low enough to underflow and silently destroy the
/// residual. It is not needed either. The kept part is `+/-0`, so its least
/// significant digit is even, and the lost part is nonzero because `x` is not
/// zero, which leaves only the halfway comparison of `|x|` against `2^n`.
///
/// That comparison never needs `2^n` to be built, because only two cases exist:
///
///   `n == e`  `|x|` lies in `[2^e, 2^(e+1))`, so it is at or above halfway, and
///             exactly at it when `x` is a power of two
///   `n > e`   `|x| < 2^(e+1) <= 2^n`, so every value is strictly below halfway
///
/// so it reduces to comparing two exponents the caller already holds, plus a
/// power-of-two test. The result is either `+/-0` or `+/-2^(n+1)`.
template <RM rm, std::floating_point T>
inline bit_float<T> round_all_lost(bit_float<T> x, exp_t e, exp_t n) {
    MPFX_DEBUG_ASSERT(!x.is_nar(), "cannot round NaN or Inf");
    MPFX_DEBUG_ASSERT(!x.is_zero(), "cannot round zero");
    MPFX_DEBUG_ASSERT(e == x.e(), "`e` must be the normalized exponent of `x`");
    MPFX_DEBUG_ASSERT(n >= e, "not every digit is unrepresentable");

    // should we round away from zero?
    bool increment;
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        if (n > e) {
            // strictly below halfway
            increment = false;
        } else {
            // at or above halfway, and exactly at it when `x` is a power of two
            const auto c = x.c();
            const bool exact_halfway = (c & (c - 1)) == 0;
            // the kept digit is even, so a tie only rounds away for RNA
            increment = exact_halfway ? rm == RM::RNA : true;
        }
    } else if constexpr (rm == RM::RTP) {
        increment = !x.s();
    } else if constexpr (rm == RM::RTN) {
        increment = x.s();
    } else if constexpr (rm == RM::RTZ) {
        increment = false;
    } else if constexpr (rm == RM::RAZ) {
        increment = true;
    } else if constexpr (rm == RM::RTO) {
        // the kept digit is even, so round to odd increments
        increment = true;
    } else if constexpr (rm == RM::RTE) {
        // the kept digit is already even
        increment = false;
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        increment = false;
    }

    // Select the magnitude rather than branching on it. For the directed modes
    // the decision follows the sign of `x`, which is not predictable on real
    // data, and mispredicting it costs more than the whole rest of this function.
    //
    // `2^(n+1)` is encoded inline rather than by calling `make_pow2`, which
    // carries a branch of its own that would stop the compiler from turning the
    // select into a conditional move. Both alternatives below are computed
    // unconditionally; the unused one is harmless, since a shift count of zero
    // and a wrapped exponent field are both well defined.
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    const exp_t k = n + 1;
    const auto normal = static_cast<uint_t>(static_cast<uint_t>(k + params_t::BIAS) << params_t::M);
    const auto shift = static_cast<unsigned>(std::max<exp_t>(params_t::EMIN - k, 0));
    const auto subnormal = static_cast<uint_t>(params_t::IMPLICIT1 >> shift);
    const uint_t pow2 = k >= params_t::EMIN ? normal : subnormal;

    const uint_t mag = increment ? pow2 : uint_t{0};
    return bit_float<T>(static_cast<uint_t>(x.sbits() | mag));
}

/// @brief Should we round away from zero?
/// @tparam rm the rounding mode
/// @param s the sign of the value being rounded
/// @param i the truncated significand, whose low bit is the last kept digit
/// @param r the residual, nonzero, with magnitude below one and the sign `s`
///
/// The residual summarizes every lost digit: its magnitude against one half
/// gives the three-way halfway comparison that `RoundRS` encodes in the
/// integer formulation, and the low bit of `i` is the last kept digit. Note
/// `i & 1` reads that digit correctly for negative `i` as well, since two's
/// complement negation preserves the low bit.
template <RM rm, std::floating_point T, signed_integral I>
inline bool round_increment_scaled(bool s, I i, T r) {
    // case split on rounding mode
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        // compare the residual against the halfway point
        const T ar = std::fabs(r);
        if (ar != static_cast<T>(0.5)) {
            return ar > static_cast<T>(0.5);
        }

        // exactly halfway
        if constexpr (rm == RM::RNA) {
            // away from zero
            return true;
        } else {
            // to even => increment if the last kept digit is odd
            return i & 1;
        }
    } else if constexpr (rm == RM::RTP) {
        // round toward +infinity
        return !s;
    } else if constexpr (rm == RM::RTN) {
        // round toward -infinity
        return s;
    } else if constexpr (rm == RM::RTZ) {
        // round toward zero
        return false;
    } else if constexpr (rm == RM::RAZ) {
        // round away from zero
        return true;
    } else if constexpr (rm == RM::RTO) {
        // round to odd => increment if the last kept digit is even
        return (i & 1) == 0;
    } else if constexpr (rm == RM::RTE) {
        // round to even => increment if the last kept digit is odd
        return i & 1;
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        return false;
    }
}

/// @brief Rounds `x` at split point `n`, where `n` is strictly below the
/// normalized exponent of `x` so that at least one digit is representable.
/// @tparam rm the rounding mode
/// @tparam FieldScale scale by exponent arithmetic rather than by multiplication
///
/// `FieldScale` requires `x` to be normal, which also makes the result normal.
/// It is the fast path; the multiplying version handles subnormal inputs and is
/// kept selectable so that the cost of the exponent trick can be measured.
template <RM rm, bool FieldScale, std::floating_point T>
inline bit_float<T> round_scaled_split(bit_float<T> x, exp_t n) {
    using int_t = typename float_params<T>::int_t;

    // Move the split point onto the binary point. Every representable digit of
    // `x` is then integral and every unrepresentable digit is fractional, so
    // the scaled value lies in `[1, 2^p)`.
    const exp_t exp = n + 1;
    T y;
    if constexpr (FieldScale) {
        y = scale_bits(x, -exp).to_float();
    } else {
        y = scale_mul(x.to_float(), -exp);
    }

    // discard the unrepresentable digits
    int_t i = static_cast<int_t>(y);

    if constexpr (rm != RM::RTZ) {
        // Recover the lost digits. This is exact: the residual spans at most
        // `P` digits, all of them below the binary point.
        const T r = y - static_cast<T>(i);
        if (r != static_cast<T>(0) && round_increment_scaled<rm>(x.s(), i, r)) {
            // move one unit away from zero, which cannot leave `[1, 2^p]`
            i += x.s() ? -1 : 1;
        }
    }

    // scale back; `|i| >= 1`, so no sign is lost on the way
    const T t = static_cast<T>(i);
    if constexpr (FieldScale) {
        return scale_bits(bit_float<T>(t), exp);
    } else {
        return bit_float<T>(scale_mul(t, exp));
    }
}

/// @brief Rounds `x` at split point `n` using a round-to-integral operation.
/// @tparam rm the rounding mode
/// @tparam FieldScale scale by exponent arithmetic rather than by multiplication
///
/// The same preconditions as `round_scaled_split`. Placing the split point on the
/// binary point makes six of the eight rounding modes a single round-to-integral
/// instruction, needing neither the residual nor the parity of the last kept
/// digit. Only round-to-odd and round-to-even still need them.
///
/// Every case here is independent of the dynamic rounding mode, matching the
/// existing implementation, which only ever adds values whose sum is exactly
/// representable.
template <RM rm, bool FieldScale, std::floating_point T>
inline bit_float<T> round_scaled_split_fp(bit_float<T> x, exp_t n) {
    // move the split point onto the binary point, as in `round_scaled_split`
    const exp_t exp = n + 1;
    T y;
    if constexpr (FieldScale) {
        y = scale_bits(x, -exp).to_float();
    } else {
        y = scale_mul(x.to_float(), -exp);
    }

    // case split on rounding mode
    T t;
    if constexpr (rm == RM::RNE) {
        // ties to even on the last kept digit is exactly round-to-nearest-even
        t = round_even(y);
    } else if constexpr (rm == RM::RNA) {
        // Ties away from zero has no round-to-integral instruction, and
        // `std::round` is an out-of-line library call. RNA differs from RNE only
        // on an exact tie, so take the nearest-even result and correct that one
        // case. `y - t` is exact, since the difference spans at most `P - 1`
        // digits below the binary point, so the test for a tie is exact too. A
        // tie also implies `|y| < 2^(P-1)`, which is where halves still exist, so
        // stepping one away from zero cannot round either.
        //
        // Biasing by one half and truncating is cheaper but is only exact while
        // `p < P`: the carry into the next binade perturbs the sum by up to
        // `2^(p-P)` of an integer, which reaches one exactly at `p == P`.
        t = round_even(y);
        if (std::fabs(y - t) == static_cast<T>(0.5)) [[unlikely]] {
            t = std::trunc(y) + std::copysign(static_cast<T>(1), y);
        }
    } else if constexpr (rm == RM::RTP) {
        // scaling by a positive power of two preserves order, so rounding
        // toward positive infinity commutes with it
        t = std::ceil(y);
    } else if constexpr (rm == RM::RTN) {
        t = std::floor(y);
    } else if constexpr (rm == RM::RTZ) {
        t = std::trunc(y);
    } else if constexpr (rm == RM::RAZ) {
        t = std::copysign(std::ceil(std::fabs(y)), y);
    } else if constexpr (rm == RM::RTO || rm == RM::RTE) {
        // No round-to-integral instruction covers these two, but the result has a
        // closed form. When `y` is inexact its two candidates are `floor(y)` and
        // `floor(y) + 1`, and exactly one of them is odd, which gives
        //
        //     RTO(y) = 2 * floor(y / 2) + 1
        //     RTE(y) = 2 * ceil(floor(y) / 2)
        //
        // Computing that unconditionally and selecting it against the exact case
        // avoids both the residual and the separate probe of the last kept digit,
        // and leaves no data-dependent branch behind. Every step is exact: halving
        // an integral value cannot round, and the results stay within `[-2^p, 2^p]`.
        const T floor_y = std::floor(y);
        T inexact;
        if constexpr (rm == RM::RTO) {
            inexact = static_cast<T>(2) * std::floor(y * static_cast<T>(0.5)) + static_cast<T>(1);
        } else {
            inexact = static_cast<T>(2) * std::ceil(floor_y * static_cast<T>(0.5));
        }

        // `y == floor_y` exactly when `y` is integral, i.e. when nothing was lost
        t = y == floor_y ? y : inexact;
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        t = y;
    }

    // scale back; `|t| >= 1`, so no sign is lost on the way
    if constexpr (FieldScale) {
        return scale_bits(bit_float<T>(t), exp);
    } else {
        return bit_float<T>(scale_mul(t, exp));
    }
}

/// @brief Rounding of a `bit_float` type by scaling.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set; only `NO_FLAGS` is supported
/// @tparam FieldScale scale by exponent arithmetic rather than by multiplication
/// @tparam FpReduce reduce to an integer with a round-to-integral operation
/// rather than with a truncating cast to an integer type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
///
/// An alternative to `round` with identical results but a floating-point rather
/// than an integer formulation. See `scale_bits` for the fast path and
/// `round_all_lost` for the underflow-to-zero path. Prefer the `round_scaled`
/// and `round_scaled_fp` wrappers below.
template <RM rm, flag_mask_t FlagMask, bool FieldScale, bool FpReduce, std::floating_point T>
bit_float<T> round_scaled_impl(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;
    MPFX_STATIC_ASSERT(FlagMask == Flags::NO_FLAGS, "round_scaled does not compute flags");
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // fast path: special values (infinity, NaN)
    if (x.is_nar()) {
        return x;
    }

    // fast path: zero
    if (x.is_zero()) {
        return x;
    }

    // compute the actual split point `n`
    const exp_t e = x.e();
    const exp_t n_min = e - static_cast<exp_t>(p);
    const exp_t n_act = n.has_value() ? std::max(n_min, *n) : n_min;

    // fast path: the split point is below every digit `x` can hold, so `x` is
    // already representable. This also keeps `n_act + 1` at or above `EXPMIN`,
    // which the scaling relies on. Only reachable for subnormal `x`.
    if (n_act < params_t::EXPMIN) {
        return x;
    }

    // no digit is representable
    if (n_act >= e) {
        return round_all_lost<rm>(x, e, n_act);
    }

    // Subnormals cannot be scaled by exponent arithmetic, and neither can the
    // subnormal results they produce, so they take the multiplying path.
    if constexpr (FieldScale) {
        if (x.ebits() != 0) {
            if constexpr (FpReduce) {
                return round_scaled_split_fp<rm, true>(x, n_act);
            } else {
                return round_scaled_split<rm, true>(x, n_act);
            }
        }
    }
    if constexpr (FpReduce) {
        return round_scaled_split_fp<rm, false>(x, n_act);
    } else {
        return round_scaled_split<rm, false>(x, n_act);
    }
}

/// @brief Rounding of a `bit_float` type by scaling, dispatching on a runtime
/// rounding mode. Prefer the `round_scaled` and `round_scaled_fp` wrappers.
template <flag_mask_t FlagMask, bool FieldScale, bool FpReduce, std::floating_point T>
bit_float<T> round_scaled_impl(bit_float<T> x, prec_t p, std::optional<exp_t> n, RM rm) {
    switch (rm) {
    case RM::RNE:
        return round_scaled_impl<RM::RNE, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RNA:
        return round_scaled_impl<RM::RNA, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RTP:
        return round_scaled_impl<RM::RTP, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RTN:
        return round_scaled_impl<RM::RTN, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RTZ:
        return round_scaled_impl<RM::RTZ, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RAZ:
        return round_scaled_impl<RM::RAZ, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RTO:
        return round_scaled_impl<RM::RTO, FlagMask, FieldScale, FpReduce>(x, p, n);
    case RM::RTE:
        return round_scaled_impl<RM::RTE, FlagMask, FieldScale, FpReduce>(x, p, n);
    default:
        MPFX_DEBUG_ASSERT(false, "round_scaled: invalid rounding mode");
        return x; // default return to avoid warnings
    }
}

/// @brief Rounding of a `bit_float` type by scaling and truncating with a cast
/// to an integer type. See `round_scaled_impl`.
template <RM rm, flag_mask_t FlagMask = Flags::NO_FLAGS, bool FieldScale = true, std::floating_point T>
bit_float<T> round_scaled(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    return round_scaled_impl<rm, FlagMask, FieldScale, false>(x, p, n);
}

/// @brief Rounding of a `bit_float` type by scaling and truncating with a cast
/// to an integer type. See `round_scaled_impl`.
template <flag_mask_t FlagMask = Flags::NO_FLAGS, bool FieldScale = true, std::floating_point T>
bit_float<T> round_scaled(bit_float<T> x, prec_t p, std::optional<exp_t> n, RM rm) {
    return round_scaled_impl<FlagMask, FieldScale, false>(x, p, n, rm);
}

/// @brief Rounding of a `bit_float` type by scaling and rounding to integral.
/// See `round_scaled_impl` and `round_scaled_split_fp`.
template <RM rm, flag_mask_t FlagMask = Flags::NO_FLAGS, bool FieldScale = true, std::floating_point T>
bit_float<T> round_scaled_fp(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    return round_scaled_impl<rm, FlagMask, FieldScale, true>(x, p, n);
}

/// @brief Rounding of a `bit_float` type by scaling and rounding to integral.
/// See `round_scaled_impl` and `round_scaled_split_fp`.
template <flag_mask_t FlagMask = Flags::NO_FLAGS, bool FieldScale = true, std::floating_point T>
bit_float<T> round_scaled_fp(bit_float<T> x, prec_t p, std::optional<exp_t> n, RM rm) {
    return round_scaled_impl<FlagMask, FieldScale, true>(x, p, n, rm);
}

} // namespace experimental

namespace {

/// @brief Encodes the result of rounding as a double-precision
/// floating-point number. This is an optimized version of `make_float<double>`
/// which assumes that `c` is either 0 or has precision exactly `P`.
template <prec_t P, unsigned_integral T>
double encode(bool s, exp_t e, T c) {
    using FP = float_params<double>::params; // double precision

    // for encoding we need to ensure that we have 53 bits of precision
    // we cannot lose bits since we guarded against too much precision,
    // i.e., `c` has at most 63 bits of precision
    uint64_t u;
    if constexpr (P > FP::P) {
        // `c` has more than 53 bits of precision
        static constexpr prec_t shift_p = P - FP::P;
        static constexpr T excess_mask = bitmask<T>(shift_p);
        MPFX_DEBUG_ASSERT((c & excess_mask) == 0, "shifting off digits");
        c >>= shift_p;
        u = static_cast<uint64_t>(c);
    } else if constexpr (P < FP::P) {
        // `c` has less than 53 bits of precision
        static constexpr prec_t shift_p = FP::P - P;
        u = static_cast<uint64_t>(c);
        u <<= shift_p;
    } else {
        // `c` has exactly 53 bits of precision
        u = static_cast<uint64_t>(c);
    }

    // encode exponent and mantissa
    uint64_t ebits, mbits;
    if (c == 0) [[unlikely]] {
        // zero result
        ebits = 0;
        mbits = 0;
    } else if (e < FP::EMIN) [[unlikely]] {
        // subnormal result
        const exp_t shift = FP::EMIN - e;
        ebits = 0;
        mbits = u >> shift;
    } else {
        // normal result - most common case
        ebits = e + FP::BIAS;
        mbits = u & FP::MMASK;
    }

    // repack the result
    const uint64_t b = (ebits << FP::M) | mbits;
    const double r = std::bit_cast<double>(b);
    return s ? -r : r;
}

/// @brief Should we increment to round?
/// @tparam T the type of the significand
/// @param s sign
/// @param c_kept current significand
/// @param c_lost lost significand bits
/// @param p_lost number of lost precision bits
/// @param rm rounding mode
/// @param overshiftp are we overshifting all digits?
/// @return should we increment the significand?
template <unsigned_integral T>
inline bool round_increment(bool s, T c_kept, T c_lost, prec_t p_lost, RM rm, bool overshiftp) {
    MPFX_DEBUG_ASSERT(p_lost > 0, "we must have lost precision");

    // case split on rounding mode
    switch (rm) {
        case RM::RNE:
        case RM::RNA: {
            // nearest rounding modes - factor out common logic
            // Compute rounding bit: -1 (below halfway), 0 (exactly halfway), 1 (above halfway)
            const T halfway = static_cast<T>(1) << (p_lost - 1);
            if (overshiftp || c_lost != halfway) [[likely]] {
                // increment if above halfway and not overshifting
                return !overshiftp && c_lost > halfway;
            }

            // exactly at halfway - tie-breaking
            if (rm == RM::RNE) {
                // ties to even: increment if LSB is odd
                return (c_kept >> p_lost) & 0x1;
            } else {
                // ties away from zero: always increment
                return true;
            }
        }
        case RM::RTP:
            // round toward +infinity
            return !s;
        case RM::RTN:
            // round toward -infinity
            return s;
        case RM::RTZ:
            // round toward zero
            return false;
        case RM::RAZ:
            // round away from zero
            return true;
        case RM::RTO:
            // round to odd => increment if LSB is even
            return ((c_kept >> p_lost) & 0x1) == 0;
        case RM::RTE:
            // round to even => increment if LSB is odd
            return (c_kept >> p_lost) & 0x1;
        default:
            MPFX_DEBUG_ASSERT(false, "unreachable");
            return false;
    }
}


/// @brief Finalizes the rounding procedure.
/// @tparam P the precision of the significand `c`
/// @tparam FlagMask the mask of flags to set
/// @param s sign
/// @param e normalized exponent
/// @param c integer significand
/// @param p precision to keep
/// @param n optional minimum normalized exponent for subnormalization
/// @param rm rounding mode
/// @return the correctly rounded result as a `double`
template <prec_t P, unsigned_integral T, flag_mask_t FlagMask>
double round_finalize(bool s, exp_t e, T c, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = float_params<double>::params; // double precision
    static constexpr size_t MAX_C_WIDTH = 8 * sizeof(T) - 1; // -1 to tolerate a carry
    MPFX_STATIC_ASSERT(P <= MAX_C_WIDTH, "mantissa is too large");

    // which flags to check
    static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_BEFORE = FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_AFTER = FlagMask & Flags::UNDERFLOW_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;

    // `MPFX_ASSERT` builds its message by concatenation, so the operand must be
    // a string rather than something streamed
    MPFX_DEBUG_ASSERT(p <= FP::P, "cannot keep the requested precision: " + std::to_string(p));

    if (c == 0) [[unlikely]] {
        // fast path: zero value
        // raise both tiny flags
        if constexpr (CHECK_TINY_BEFORE) {
            flags.set_tiny_before_rounding();
        }
        if constexpr (CHECK_TINY_AFTER) {
            flags.set_tiny_after_rounding();
        }

        // return +/-0
        return s ? -0.0 : 0.0;
    }

    prec_t p_kept = p;        // actual precision kept
    bool overshiftp = false;  // are all digits insignificant and non-adjacent to n?
    bool tiny_before = false; // was the value tiny before rounding?
    bool tiny_after = false;  // was the value tiny after rounding?

    // handle possible subnormalization
    if (n.has_value()) {
        // compute the minimum normalized exponent
        const exp_t emin = *n + static_cast<exp_t>(p);
        const exp_t eoffset = emin - e;

        if (eoffset > 0) {
            // our precision is limited by subnormalization
            const prec_t shift = static_cast<prec_t>(eoffset);

            // we are definitely tiny before rounding
            tiny_before = true;
            if constexpr (CHECK_TINY_BEFORE) {
                flags.set_tiny_before_rounding();
            }

            // check for tininess after rounding
            if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
                // check for the easy case of tininess after rounding

                // significand of the largest representable value in a binade
                const T cutoff = bitmask<T>(p) << (P - p);

                // tiny if we are below: 1.111...111 x 2^(emin-1)
                // if not set, we are in the hard case and need to check
                // for tininess after splitting the significand
                tiny_after = shift > 1 || (c <= cutoff);

                // set tiny after rounding flag if tiny before
                if constexpr (CHECK_TINY_AFTER) {
                    if (tiny_after) {
                        flags.set_tiny_after_rounding();
                    }
                }
            }

            // "overshift" is set if we shift more than p bits
            overshiftp = shift > p; // set overshift flag
            p_kept = overshiftp ? 0 : p - shift; // precision cannot be non-positive
            e = overshiftp ? *n : e; // set exponent for subnormalization
        }
    }

    // extract the lost digits
    const prec_t p_lost = p_kept < P ? P - p_kept : 0;
    const T c_mask = bitmask<T>(p_lost);
    const T c_lost = c & c_mask;

    // check if we rounded off any significant digits
    if (c_lost != 0) {
        // slow path: inexact result
        MPFX_DEBUG_ASSERT(p_lost > 0, "we must have lost precision");
        T c_kept = c & ~c_mask; // mask off lost digits

        // check the hard case for tiny after rounding
        if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
            if (tiny_before && !tiny_after) [[unlikely]] {
                // we are just below 2^emin but above the cutoff value
                MPFX_DEBUG_ASSERT(n.has_value(), "n must be set");
                MPFX_DEBUG_ASSERT(p_kept < P, "must have kept at least one digit");
                MPFX_DEBUG_ASSERT(p_lost > 1, "must have lost at least 2 digits");
                MPFX_DEBUG_ASSERT(!overshiftp, "must not have overshifted all digits");

                // need to check if we round to 2^emin (unbounded exponent)
                // by rounding with a split that is one digit lower
                const T one = static_cast<T>(1) << (p_lost - 1); // dummy value to indicate oddness
                const T c_half_mask = bitmask<T>(p_lost - 1);
                const T c_lost_half = c_lost & c_half_mask;
                tiny_after = !round_increment(s, one, c_lost_half, p_lost - 1, rm, false);

                // set tiny after rounding flag if tiny before
                if constexpr (CHECK_TINY_AFTER) {
                    if (tiny_after) {
                        flags.set_tiny_after_rounding();
                    }
                }
            }
        }

        // should we increment?
        if (round_increment(s, c_kept, c_lost, p_lost, rm, overshiftp)) {
            // size of the increment
            const T one = static_cast<T>(1) << p_lost;

            // apply increment
            c_kept += one;

            // check if we need to carry
            static constexpr T overflow_mask = static_cast<T>(1) << P;
            if (c_kept >= overflow_mask) [[unlikely]] {
                // increment caused carry
                e += 1;
                c_kept >>= 1;
                if constexpr (CHECK_CARRY) {
                    if (!tiny_before) {
                        flags.set_carry();
                    }
                }
            }
        }

        // final significand after rounding
        c = c_kept;

        // set the underflow flags
        if constexpr (CHECK_UNDERFLOW_BEFORE) {
            if (tiny_before) {
                flags.set_underflow_before_rounding();
            }
        }
        if constexpr (CHECK_UNDERFLOW_AFTER) {
            if (tiny_after) {
                flags.set_underflow_after_rounding();
            }
        }

        // set inexact flag
        if constexpr (CHECK_INEXACT) {
            flags.set_inexact();
        }
    }

    return encode<P>(s, e, c);
}

} // anonymous namespace

/// @brief Optimized rounding to round a floating-point number of type `T`
/// to a value of the same type with target precision `p` and first
/// unrepresentable digit `n`. Rounding happens in `T`'s own container, so a
/// `float` argument is not widened to `double` (which would double-round).
template<flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
T round(T x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    return experimental::round<FlagMask>(bit_float<T>(x), p, n, rm).to_float();
}

/// @brief Optimized rounding to round `m * 2^exp`
/// to a double-precision floating-point number with target precision `p`
/// and first unrepresentable digit `n`.
template<flag_mask_t FlagMask = Flags::ALL_FLAGS, signed_integral T>
double round(T m, exp_t exp, prec_t p, const std::optional<exp_t>& n, RM rm) {
    static constexpr T MIN_VAL = std::numeric_limits<T>::min();
    static constexpr prec_t PREC = 8 * sizeof(T) - 1; // -1 due to conversion to unsigned
    using U = make_unsigned_t<T>;

    // Decode `m` into sign-magnitude
    bool s;
    U c;
    if (m < 0) {
        s = true;
        if (m == MIN_VAL) {
            // special decode to ensure 63 bits of precision
            c = static_cast<U>(1) << (PREC - 1);
            exp += 1;
        } else {
            // normal decode
            c = static_cast<U>(-m);
        }
    } else {
        s = false;
        c = static_cast<U>(m);
    }

    // normalize the input
    const auto lz = PREC - bit_width(c);
    c <<= lz;
    exp -= lz;

    // calculate normalized exponent
    const exp_t e = exp + (PREC - 1);

    // finalize rounding (mantissa has precision 63)
    return round_finalize<PREC, U, FlagMask>(s, e, c, p, n, rm);
}

} // namespace mpfx
