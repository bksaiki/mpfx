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

/// @brief Rounds `x` at split point `n`, where `n` is strictly below the
/// normalized exponent of `x` so that at least one digit is representable.
/// @tparam rm the rounding mode
///
/// Placing the split point on the binary point turns six of the eight rounding
/// modes into a single round-to-integral instruction, and the remaining two into
/// a closed form, so none of them needs the residual or a probe of the last kept
/// digit.
///
/// Every case here is independent of the dynamic rounding mode, matching the
/// existing implementation, which only ever adds values whose sum is exactly
/// representable.
///
/// Subnormals are handled by biasing rather than by a separate path. Exponent
/// arithmetic cannot scale a subnormal directly - its exponent field is zero, so
/// adding to it would fabricate an implicit bit - and neither can it produce the
/// subnormal results that subnormal inputs give rise to. Multiplying instead is
/// correct but costs about 30 ns per operation, because the hardware takes a
/// microcode assist for a multiply with a denormal operand and again for any
/// operation with a denormal result.
///
/// Adding `IMPLICIT1` to the *encoding* of a subnormal yields exactly
/// `x + sgn(x) * 2^EMIN`, which is normal, and the encoding stays linear in the
/// value across the boundary, so subtracting it again afterwards recovers the
/// result. Both steps are integer arithmetic on the encoding, so no denormal ever
/// reaches the floating-point unit. The bias in the scaled domain is
/// `2^(EMIN-exp)`, an *even* integer, and every rounding mode here commutes with
/// adding an even integer of the same sign - including the parity that RTO and RTE
/// depend on and the ties that RNE breaks - so there is nothing to undo in
/// between. For a normal `x` the bias is zero and the whole thing vanishes, which
/// is why one code path serves both.
/// @brief A rounded value together with what the status flags need to know about
/// how it was reached. `inexact` falls out of a comparison the scaled form already
/// has in hand, and constant-folds away when no flag asks for it.
template <std::floating_point T>
struct ScaledResult {
    bit_float<T> value;
    bool inexact; // digits were discarded
};

template <RM rm, bool Biased, std::floating_point T>
inline ScaledResult<T> round_scaled_split(bit_float<T> x, exp_t n) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    // Move the split point onto the binary point. Every representable digit of
    // `x` is then integral and every unrepresentable digit is fractional, so the
    // scaled value lies in `[1, 2^p)`, offset by the bias when `x` is subnormal.
    const exp_t exp = n + 1;
    static constexpr uint_t bias = Biased ? static_cast<uint_t>(params_t::IMPLICIT1) : uint_t{0};
    const bit_float<T> xb(static_cast<uint_t>(x.to_bits() + bias));
    const T y = scale_bits(xb, -exp).to_float();

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
    } else if constexpr (rm == RM::RTO) {
        // No round-to-integral instruction covers round to odd or round to even,
        // but both have a closed form. When `y` is inexact its two candidates are
        // `floor(y)` and `floor(y) + 1`, exactly one of which is odd, so
        //
        //     RTO(y) = 2 * floor(y / 2) + 1
        //
        // Computing that unconditionally and selecting it against the exact case
        // avoids both the residual and a separate probe of the last kept digit, and
        // leaves no data-dependent branch behind. Every step is exact: halving an
        // integral value cannot round, and the result stays within `[-2^p, 2^p]`.
        const T floor_y = std::floor(y);
        const T odd = static_cast<T>(2) * std::floor(y * static_cast<T>(0.5)) + static_cast<T>(1);

        // `y == floor_y` exactly when `y` is integral, i.e. when nothing was lost
        t = y == floor_y ? y : odd;
    } else if constexpr (rm == RM::RTE) {
        // the even candidate of the same pair, by the same argument as RTO above
        //
        //     RTE(y) = 2 * ceil(floor(y) / 2)
        const T floor_y = std::floor(y);
        const T even = static_cast<T>(2) * std::ceil(floor_y * static_cast<T>(0.5));
        t = y == floor_y ? y : even;
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        t = y;
    }

    // `t` is integral, so `y` differs from it exactly when digits were discarded.
    // This survives the bias, which shifts `y` and `t` by the same integer.
    const bool inexact = y != t;

    // scale back and remove the bias; `|t| >= 1`, so no sign is lost on the way
    const bit_float<T> ub = scale_bits(bit_float<T>(t), exp);
    if constexpr (Biased) {
        return {bit_float<T>(static_cast<uint_t>(ub.to_bits() - bias)), inexact};
    } else {
        return {ub, inexact};
    }
}

/// @brief Whether rounding `x` at `n_min` with an unbounded exponent range still
/// lands below `2^emin`.
/// @tparam rm the rounding mode
/// @param x the value to round, whose normalized exponent must be `emin - 1`
/// @param n_min the unsubnormalized split point, at or above `EXPMIN`
/// @param emin the smallest normalized exponent of the emulated format
///
/// The binade directly below `2^emin` is the only one where tininess after rounding
/// can differ from tininess before it: rounding raises the normalized exponent by at
/// most one, so anything lower stays tiny however it rounds. Within that binade the
/// result lies in `[2^(emin-1), 2^emin]`, so the only way not to be tiny is to carry
/// exactly onto `2^emin`. That asks two much cheaper questions than a second rounding
/// does - whether `|x|` lies in the last grid cell of the binade, and whether this
/// mode rounds up out of it.
///
/// Both are answered on the encoding, which is linear in the value within a binade
/// and across the whole subnormal range, so one path serves normal and subnormal `x`
/// and no denormal reaches the floating-point unit.
template <RM rm, std::floating_point T>
inline bool tiny_after_unbounded(bit_float<T> x, exp_t n_min, exp_t emin) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    MPFX_DEBUG_ASSERT(!x.is_nar() && !x.is_zero(), "cannot probe NaN, Inf, or zero");
    MPFX_DEBUG_ASSERT(x.e() + 1 == emin, "`x` must lie in the binade below `2^emin`");
    MPFX_DEBUG_ASSERT(n_min >= params_t::EXPMIN, "split point is below the encoding");

    // how many encoding bits lie below the rounding grid
    const exp_t shift = n_min + 1 - x.exp();
    if (shift <= 0) {
        // every value the container can hold is already on the grid, so exact
        return true;
    }

    const uint_t mag = static_cast<uint_t>(x.to_bits() & ~params_t::SMASK);
    const uint_t step = static_cast<uint_t>(uint_t{1} << shift);
    const uint_t lost = static_cast<uint_t>(mag & (step - 1));
    const uint_t pow2 = bit_float<T>::make_pow2(emin).to_bits();

    // `lost == 0` is exact, and otherwise only the last cell can carry
    if (lost == 0 || static_cast<uint_t>((mag - lost) + step) != pow2) {
        return true;
    }

    // Does this mode round up out of the last cell? The two candidates are `2^emin`,
    // whose last kept digit is even, and its predecessor, whose kept digits are all
    // ones and so odd.
    bool up;
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        // a tie goes to `2^emin` for both: it is the even candidate, and it is also
        // the one away from zero
        up = lost >= (step >> 1);
    } else if constexpr (rm == RM::RTP) {
        up = !x.s();
    } else if constexpr (rm == RM::RTN) {
        up = x.s();
    } else if constexpr (rm == RM::RTZ) {
        up = false;
    } else if constexpr (rm == RM::RAZ) {
        up = true;
    } else if constexpr (rm == RM::RTO) {
        // the predecessor is the odd candidate
        up = false;
    } else if constexpr (rm == RM::RTE) {
        // `2^emin` is the even candidate
        up = true;
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        up = false;
    }
    return !up;
}

/// @brief The general path of `round_scaled`: handles every input.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
///
/// `round_scaled` peels off the one shape of input that real data is made of
/// and sends everything else here, so that its own body stays small enough for
/// the compiler to inline into callers' loops. This function is complete on its
/// own - it re-derives everything from `x` - and stays out of line: the cost of
/// a call only falls on inputs that already take the expensive paths.
template <RM rm, flag_mask_t FlagMask, std::floating_point T>
bit_float<T> round_scaled_general(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;

    // which flags to check
    static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_BEFORE = FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_AFTER = FlagMask & Flags::UNDERFLOW_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // fast path: special values (infinity, NaN)
    if (x.is_nar()) {
        return x;
    }

    // fast path: zero
    if (x.is_zero()) {
        // zero counts as tiny, both before and after
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
    const bool tiny_before = e < emin;
    if constexpr (CHECK_TINY_BEFORE) {
        if (tiny_before) {
            flags.set_tiny_before_rounding();
        }
    }

    // fast path: the split point is below every digit `x` can hold, so `x` is
    // already representable. This also keeps `n_act + 1` at or above `EXPMIN`,
    // which the scaling relies on. Only reachable for subnormal `x`.
    if (n_act < params_t::EXPMIN) {
        // exact, so tininess after rounding matches tininess before
        if constexpr (CHECK_TINY_AFTER) {
            if (tiny_before) {
                flags.set_tiny_after_rounding();
            }
        }
        return x;
    }

    // no digit is representable
    ScaledResult<T> r;
    if (n_act >= e) {
        // `x` is nonzero and every digit was discarded, so this is always inexact
        r = {round_all_lost<rm>(x, e, n_act), true};
    } else
    // A subnormal is biased into the normal range first; see `round_scaled_split`.
    // Biasing unconditionally would remove this branch, and was measured: it costs
    // about 20% on every input, and only pays off once more than roughly one input
    // in sixteen is subnormal, where the misprediction starts to dominate. Container
    // subnormals need `|x| < 2^EMIN`, which emulating a narrower format in a wider
    // container never reaches, so the branch stays.
    if (x.ebits() != 0) {
        r = round_scaled_split<rm, false>(x, n_act);
    } else {
        r = round_scaled_split<rm, true>(x, n_act);
    }

    // set inexact flag if requested
    if constexpr (CHECK_INEXACT) {
        if (r.inexact) {
            flags.set_inexact();
        }
    }

    if (tiny_before) {
        // underflow is tininess together with inexactness
        if constexpr (CHECK_UNDERFLOW_BEFORE) {
            if (r.inexact) {
                flags.set_underflow_before_rounding();
            }
        }

        if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
            // Tiny after rounding asks whether the value would still fall below the
            // smallest normal *were the exponent range unbounded*, so it cannot be
            // read off `r`: that was rounded at the subnormalized split `n_act`,
            // which is coarser and can reach `2^emin` when an unbounded rounding
            // would not. Rounding again at `n_min` answers the question as asked.
            //
            // Only the binade directly below `2^emin` can reach it, since rounding
            // raises the normalized exponent by at most one, so everything lower is
            // tiny without further work. An exact result keeps its magnitude and is
            // tiny too, and if `n_min` sits below every digit `x` holds then the
            // unbounded rounding is exact.
            bool tiny_after;
            if (!r.inexact || e < emin - 1 || n_min < params_t::EXPMIN) {
                tiny_after = true;
            } else {
                // Reaching here forces `e == emin - 1`: the guard above rules out
                // anything lower and `tiny_before` rules out anything higher, so
                // `x` lies in the one binade where the two tininess answers can
                // differ. See `tiny_after_unbounded`.
                tiny_after = tiny_after_unbounded<rm>(x, n_min, emin);
            }
            if (tiny_after) {
                if constexpr (CHECK_TINY_AFTER) {
                    flags.set_tiny_after_rounding();
                }
                if constexpr (CHECK_UNDERFLOW_AFTER) {
                    if (r.inexact) {
                        flags.set_underflow_after_rounding();
                    }
                }
            }
        }
    } else if constexpr (CHECK_CARRY) {
        // A rounding that changes the normalized exponent must land exactly on a
        // power of two, so a zero significand together with a move away from zero
        // says so.
        //
        // The order of the two tests matters more than either test does. A rounding
        // moves away from zero about half the time, so leading with that gives the
        // branch predictor a coin flip and costs five times the rest of this
        // function put together. Landing on a power of two is rare, so leading with
        // that keeps the branch almost always false and the comparison behind it
        // almost never runs.
        if (r.value.mbits() == 0 && r.value.compare_mag(x) > 0) {
            flags.set_carry();
        }
    }

    return r.value;
}

/// @brief Rounding of a `bit_float` type by scaling.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
///
/// An alternative to `round` with identical results but a floating-point rather
/// than an integer formulation. See `scale_bits` for the fast path and
/// `round_all_lost` for the underflow-to-zero path.
///
/// This body handles only the common shape of input - finite, normal, above the
/// subnormalization point, with at least one representable digit - and delegates
/// everything else to `round_scaled_general`. The split is for the inliner as
/// much as for the reader: rounding is a few instructions per call, so whether
/// the call itself dissolves into the surrounding loop is worth more than any
/// single instruction in it, and only a body this small dissolves. The delegate
/// stays a real call and its cost falls only on the rare shapes.
template <RM rm, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
bit_float<T> round_scaled(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;
    MPFX_DEBUG_ASSERT(p >= 1, "target precision must be at least one digit");
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // Zero, subnormal, infinity, and NaN all leave through the general path.
    // Zero and subnormal `x` have a zero exponent field, whose decrement wraps
    // to the top, and Inf/NaN sit exactly at `EMASK`, so one unsigned
    // comparison covers all four.
    const uint_t ebits = x.ebits();
    if (static_cast<uint_t>(ebits - 1) >= static_cast<uint_t>(params_t::EMASK) - 1) {
        return round_scaled_general<rm, FlagMask>(x, p, n);
    }

    // the split point of the unsubnormalized format; `x` is normal, so this
    // needs no `countl_zero`. `p >= 1` keeps `n_min` strictly below `e`, which
    // the split at the bottom relies on.
    const exp_t e = static_cast<exp_t>(ebits >> params_t::M) - params_t::BIAS;
    const exp_t n_min = e - static_cast<exp_t>(p);

    // Subnormalization wins the `max` exactly when `n_min < *n`, which is also
    // exactly when `x` is tiny before rounding, so one comparison rules out
    // every subnormal-format and underflow concern at once.
    if (n.has_value() && n_min < *n) {
        return round_scaled_general<rm, FlagMask>(x, p, n);
    }

    // fast path: the split point is below every digit `x` can hold, so `x` is
    // already representable. Exact and not tiny, so no flag can raise.
    if (n_min < params_t::EXPMIN) {
        return x;
    }

    // `x` is normal with `EXPMIN <= n_min < e`: exactly the split's fast shape
    const ScaledResult<T> r = round_scaled_split<rm, false>(x, n_min);

    // set inexact flag if requested
    if constexpr (CHECK_INEXACT) {
        if (r.inexact) {
            flags.set_inexact();
        }
    }

    // not tiny, so only the carry flag remains; see `round_scaled_general` for
    // why the power-of-two test comes first
    if constexpr (CHECK_CARRY) {
        if (r.value.mbits() == 0 && r.value.compare_mag(x) > 0) {
            flags.set_carry();
        }
    }

    return r.value;
}

/// @brief Rounding of a `bit_float` type by scaling, dispatching on a runtime
/// rounding mode. See the compile-time overload above.
template <flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
bit_float<T> round_scaled(bit_float<T> x, prec_t p, std::optional<exp_t> n, RM rm) {
    switch (rm) {
    case RM::RNE:
        return round_scaled<RM::RNE, FlagMask>(x, p, n);
    case RM::RNA:
        return round_scaled<RM::RNA, FlagMask>(x, p, n);
    case RM::RTP:
        return round_scaled<RM::RTP, FlagMask>(x, p, n);
    case RM::RTN:
        return round_scaled<RM::RTN, FlagMask>(x, p, n);
    case RM::RTZ:
        return round_scaled<RM::RTZ, FlagMask>(x, p, n);
    case RM::RAZ:
        return round_scaled<RM::RAZ, FlagMask>(x, p, n);
    case RM::RTO:
        return round_scaled<RM::RTO, FlagMask>(x, p, n);
    case RM::RTE:
        return round_scaled<RM::RTE, FlagMask>(x, p, n);
    default:
        MPFX_DEBUG_ASSERT(false, "round_scaled: invalid rounding mode");
        return x; // default return to avoid warnings
    }
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
