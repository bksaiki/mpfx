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

/// Shared primitives for the rounding implementations below. Both `reference` and
/// `round_scaled` build on these, so a differential test between the two does not
/// exercise them independently - they are covered by the exhaustive sweeps instead.
/// Primitives shared by `round_bits` and `round_scaled`. This is deliberately
/// small: a differential test between those two cannot check anything in here,
/// since both sides use it. It is covered by the exhaustive sweeps instead.
namespace round_internal {

/// @brief Should we increment to round, for the directed modes?
/// @tparam rm the rounding mode
/// @tparam T the floating-point container type
/// @param hi the high part of the split significand
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

/// @brief Whether rounding `x` at `n_min` with an unbounded exponent range still
/// lands below `2^emin`.
/// @tparam rm the rounding mode
/// @tparam T the floating-point container type
/// @param x the value to round, whose normalized exponent must be `emin - 1`
/// @param n_min the unsubnormalized split point, at or above `EXPMIN`
/// @param emin the smallest normalized exponent of the emulated format
/// @return is the unbounded rounding still tiny?
///
/// The result lies in `[2^(emin-1), 2^emin]`, so the only escape is carrying exactly
/// onto `2^emin`. Everything below is answered on the encoding, which is linear in
/// the value across the subnormal range and on into the normal one - so the last
/// cell is found by integer arithmetic even when it ends at `2^EMIN`, and one path
/// serves normal and subnormal `x` alike.
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

    // Does this mode round up out of the last cell? The two candidates are `2^emin`
    // and its predecessor, the kept part, whose digits are all ones and so odd.
    bool up;
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        // a tie goes to `2^emin` for both: it is the even candidate, and it is also
        // the one away from zero
        up = lost >= (step >> 1);
    } else {
        const bit_float<T> hi(static_cast<uint_t>(x.sbits() | (mag - lost)));
        up = round_internal::round_increment_directed<rm>(hi, n_min);
    }
    return !up;
}

/// @brief Raises the tiny- and underflow-after-rounding flags for an `x` that was
/// tiny before rounding, and is neither zero nor NaN/Inf.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the value that was rounded
/// @param e the normalized exponent of `x`
/// @param n_min the unsubnormalized split point, `e - p`
/// @param emin the smallest normalized exponent of the emulated format
/// @param inexact whether the rounding discarded digits
///
/// Tininess after rounding is asked of the *unbounded* rounding at `n_min`, not the
/// subnormalized one the caller performed. Exact results stay tiny, as does anything
/// below the binade under `2^emin`; the rest is `tiny_after_unbounded`.
template <RM rm, flag_mask_t FlagMask, std::floating_point T>
inline void set_tiny_after(bit_float<T> x, exp_t e, exp_t n_min, exp_t emin, bool inexact) {
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_AFTER = FlagMask & Flags::UNDERFLOW_AFTER_ROUNDING_FLAG;
    if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
        using params_t = typename bit_float<T>::params_t;
        const bool tiny_after = !inexact || e < emin - 1 || n_min < params_t::EXPMIN
                             || tiny_after_unbounded<rm>(x, n_min, emin);
        if (tiny_after) {
            if constexpr (CHECK_TINY_AFTER) {
                flags.set_tiny_after_rounding();
            }
            if constexpr (CHECK_UNDERFLOW_AFTER) {
                if (inexact) {
                    flags.set_underflow_after_rounding();
                }
            }
        }
    }
}

} // namespace round_internal

/// Rounds by manipulating the `bit_float` encoding directly, never decoding it into
/// a significand triple. `round_scaled` is what production code uses; this is the
/// oracle the test suite rounds against, since a differential test needs two
/// genuinely different algorithms. It is the better oracle of the two available:
/// unlike `round_reference` it can express NaN, infinity, and signed zero.
///
/// Note that unlike `round_scaled`, this path is *not* independent of the host
/// rounding mode: its increment is a real floating-point add, so a carry that
/// overflows to infinity is decided by the host mode. Differential tests must run
/// under the default mode.
namespace round_bits {

/// @brief Should we increment to round, for the nearest modes?
/// @tparam rm the rounding mode
/// @tparam T the floating-point container type
/// @param hi the high part of the split significand
/// @param n the split point
/// @param rs the low part of the significand, in a round-sticky scheme
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

/// @brief Rounds `x` at split point `n` in the integer domain.
/// @tparam rm the rounding mode
/// @tparam T the floating-point container type
/// @param x the value to round, neither NaN nor Inf
/// @param n the split point
/// @param out the rounded value
/// @return were any digits discarded?
///
/// The value leaves through `out` rather than as a returned pair: a struct returned
/// from several `return` sites defeats the compiler's scalar replacement, and the
/// spill costs more than the rounding on the callers' hot paths.
template <RM rm, std::floating_point T>
inline bool round_split(bit_float<T> x, exp_t n, bit_float<T>& out) {
    bool increment;
    if constexpr (rm == RM::RNE || rm == RM::RNA) {
        const auto [hi, rs] = x.split_rs(n);
        out = hi;
        if (rs == RoundRS::EXACT) {
            return false;
        }
        increment = round_increment_nearest<rm>(hi, n, rs);
    } else {
        const auto [hi, sticky] = x.split_sticky(n);
        out = hi;
        if (!sticky) {
            return false;
        }
        increment = round_internal::round_increment_directed<rm>(hi, n);
    }
    if (increment) {
        out = out.next_away_zero(n + 1);
    }
    return true;
}

/// @brief The general path of `round`: handles every input.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
/// @return the rounded value
template <RM rm, flag_mask_t FlagMask, std::floating_point T>
bit_float<T> round_general(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // which flags to check
    static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_BEFORE = FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG;
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

    bit_float<T> result;
    const bool inexact = round_split<rm>(x, n_act, result);

    // fast path: nothing lost, so tininess after rounding matches tininess before
    if (!inexact) {
        if constexpr (CHECK_TINY_AFTER) {
            if (tiny_before) {
                flags.set_tiny_after_rounding();
            }
        }
        return result;
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
        round_internal::set_tiny_after<rm, FlagMask>(x, e, n_min, emin, true);
    } else if constexpr (CHECK_CARRY) {
        // A carry raises the normalized exponent, given `x` not tiny - see
        // `TestFlags.TestCarryFlag` - and can only do so by landing exactly on a
        // power of two above `x`. Testing the *significand* for that covers both
        // normal and subnormal results; a zero mantissa field alone would miss a
        // carry onto a subnormal power of two, whose field is non-zero.
        const auto c = result.c();
        if ((c & (c - 1)) == 0 && result.compare_mag(x) > 0) {
            flags.set_carry();
        }
    }

    return result;
}

/// @brief Optimized rounding of a `bit_float` type.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
/// @return the rounded value
///
/// Handles only the common shape - finite, normal, above the subnormalization
/// point, with at least one representable digit - and delegates the rest to
/// `round_general`, mirroring `round_scaled::round`; see there for why.
template <RM rm, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
bit_float<T> round(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;
    MPFX_DEBUG_ASSERT(p >= 1, "target precision must be at least one digit");
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // Zero, subnormal, infinity, and NaN all leave through the general path;
    // see `round_scaled::round` for how one comparison covers all four.
    const uint_t bits = x.to_bits();
    const uint_t ebits = static_cast<uint_t>(bits & params_t::EMASK);
    if (static_cast<uint_t>(ebits - 1) >= static_cast<uint_t>(params_t::EMASK) - 1) {
        return round_general<rm, FlagMask>(x, p, n);
    }

    // bounds on the biased exponent, built from `p` and `n` alone; these are
    // the tininess/subnormalization and exactness tests of `round_scaled::round`
    const int64_t eb = static_cast<int64_t>(ebits >> params_t::M);
    if (n.has_value() && eb < *n + static_cast<int64_t>(p) + params_t::BIAS) {
        return round_general<rm, FlagMask>(x, p, n);
    }
    if (eb < params_t::EXPMIN + static_cast<int64_t>(p) + params_t::BIAS) {
        return x;
    }

    // `x` is normal with `EXPMIN <= n_min < e`: exactly the shape the general
    // body handles on its happy path, minus only what the guards above have
    // already decided (no zero, no tiny, no subnormalized split).
    const exp_t e = static_cast<exp_t>(eb) - params_t::BIAS;
    const exp_t n_min = e - static_cast<exp_t>(p);
    bit_float<T> result;
    if (!round_split<rm>(x, n_min, result)) {
        // nothing lost, so exact, and (not being tiny) no flag can raise
        return result;
    }

    if constexpr (CHECK_INEXACT) {
        flags.set_inexact();
    }

    // Not tiny, so only the carry flag remains. `x` is normal here, so the result
    // is too, and the exponent field orders the same way as `e()` does in
    // `round_general` - no need to normalize.
    if constexpr (CHECK_CARRY) {
        if (result.ebits() > ebits) {
            flags.set_carry();
        }
    }

    return result;
}

/// @brief Optimized rounding of a `bit_float` type, dispatching on a runtime
/// rounding mode. See the compile-time overload above.
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
/// @param rm the rounding mode
/// @return the rounded value
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

} // namespace round_bits

/// The scale-and-truncate implementation - what the public `round` uses. Works in
/// the floating-point rather than the integer domain, and unlike
/// `round_bits::round` it is independent of the host rounding mode.
namespace round_scaled {

///
/// Scale-and-truncate rounding.
///
/// An alternative formulation of `round` in the floating-point domain. Scaling
/// `x` by `2^-(n+1)` puts the split point exactly on the binary point: the
/// representable digits of `x` become the integer part of the scaled value and
/// the unrepresentable digits its fraction, so rounding becomes a
/// round-to-integral operation. The scaling is exponent arithmetic on the
/// encoding, which leaves the mantissa and sign untouched and so is exact.
///

/// @brief A rounded value together with what the status flags need to know about
/// how it was reached.
/// @tparam T the floating-point container type
template <std::floating_point T>
struct RoundResult {
    bit_float<T> value;
    bool inexact; // digits were discarded
};

/// @brief Rounds `x` when every digit is unrepresentable, i.e. `n >= e`.
/// @tparam rm the rounding mode
/// @tparam T the floating-point container type
/// @param x the value to round, neither zero nor NaN/Inf
/// @param e the normalized exponent of `x`
/// @param n the actual split point, at or above `e`
/// @return `+/-0` or `+/-2^(n+1)`
///
/// Scaling would underflow here and destroy the digits it is meant to summarize,
/// and is not needed: the kept part is `+/-0`, whose last digit is even, and the
/// lost part is nonzero, so only the halfway comparison against `2^n` remains.
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
    } else {
        // the directed modes read only the sign and the last kept digit, and the
        // kept part `+/-0` supplies both: its significand is zero, so every digit
        // reads even
        increment = round_internal::round_increment_directed<rm>(bit_float<T>(x.sbits()), n);
    }

    // Select the magnitude rather than branching on it, and encode `2^(n+1)` inline
    // rather than via `make_pow2`, whose own branch would stop the compiler from
    // emitting a conditional move. The unused alternative is harmless: a zero shift
    // count and a wrapped exponent field are both well defined.
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

/// @brief Rounds `y` to an integral value, with the mode pinned rather than read
/// from the dynamic rounding state.
/// @tparam rm the rounding mode
/// @tparam T the floating-point container type
/// @param y the value to round, with `1 <= |y| <= 2^P` as the scaling guarantees
/// @return the rounded integral value
template <RM rm, std::floating_point T>
inline T round_to_integral(T y) {
    // case split on rounding mode
    if constexpr (rm == RM::RNE) {
        // ties to even on the last kept digit is exactly round-to-nearest-even
        return round_even(y);
    } else if constexpr (rm == RM::RNA) {
        // RNA differs from RNE only on an exact tie, so correct that one case. A
        // tie implies `|y| < 2^(P-1)`, where halves still exist, so both `y - t`
        // and the step away from zero are exact. Biasing by one half and truncating
        // is cheaper but only exact while `p < P`.
        const T t = round_even(y);
        if (std::fabs(y - t) == static_cast<T>(0.5)) [[unlikely]] {
            return std::trunc(y) + std::copysign(static_cast<T>(1), y);
        } else {
            return t;
        }
    } else if constexpr (rm == RM::RTP) {
        // scaling by a positive power of two preserves order, so rounding
        // toward positive infinity commutes with it
        return std::ceil(y);
    } else if constexpr (rm == RM::RTN) {
        return std::floor(y);
    } else if constexpr (rm == RM::RTZ) {
        return std::trunc(y);
    } else if constexpr (rm == RM::RAZ) {
        return std::copysign(std::ceil(std::fabs(y)), y);
    } else if constexpr (rm == RM::RTO) {
        // No instruction covers round to odd or round to even, but halving gives
        // each a closed form. `o = 2 * floor(y / 2)` is the even integer at or
        // below `y`, so `o + 1` is the odd candidate, and is `y` itself when `y`
        // is odd. Only an even `y` needs correcting, i.e. `y == o`.
        const T half = std::floor(y * static_cast<T>(0.5));
        const T o = half + half;
        return y == o ? y : o + static_cast<T>(1);
    } else if constexpr (rm == RM::RTE) {
        // The even candidate of the same pair: `t0 = 2 * roundeven(y / 2)`, which
        // is `y` itself when `y` is even. Only an odd `y` needs correcting, and it
        // is the only input landing exactly one away. `y - t0` is exact.
        const T half = round_even(y * static_cast<T>(0.5));
        const T t0 = half + half;
        return std::fabs(y - t0) == static_cast<T>(1) ? y : t0;
    } else {
        MPFX_DEBUG_ASSERT(false, "unreachable");
        return y;
    }
}

/// @brief Rounds `x` at split point `n`, with `n` strictly below the normalized
/// exponent of `x` so that at least one digit is representable.
/// @tparam rm the rounding mode
/// @tparam Biased whether `x` is subnormal and must be biased into the normal range
/// @tparam T the floating-point container type
/// @param x the value to round, neither zero nor NaN/Inf
/// @param n the split point, with `n + 1 >= EXPMIN`
/// @return the rounded value, and whether digits were discarded
///
/// Exponent arithmetic cannot scale a subnormal - its field is zero, so adding to
/// it would fabricate an implicit bit - and multiplying costs ~30 ns in denormal
/// microcode assists. So subnormals are biased instead: adding `IMPLICIT1` to the
/// *encoding* gives exactly `x + sgn(x) * 2^EMIN`, which is normal, and the encoding
/// stays linear across the boundary, so subtracting it again recovers the result.
///
/// The bias in the scaled domain is `2^(EMIN-exp)`, an *even* integer, and every
/// mode commutes with adding an even integer of the same sign - the parity RTO and
/// RTE depend on and the ties RNE breaks included - so nothing needs undoing in
/// between. For a normal `x` the bias is zero and the whole thing vanishes.
template <RM rm, bool Biased, std::floating_point T>
inline RoundResult<T> round_split(bit_float<T> x, exp_t n) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    // scale down: move the split point onto the binary point, putting the scaled
    // value in `[1, 2^p)`, offset by the bias when `x` is subnormal. One shifted
    // constant serves both directions, since the wrapped sum is the two's
    // complement either way: subtracting `d` scales by `2^-exp`, adding it by `2^exp`.
    const exp_t exp = n + 1;
    static constexpr uint_t bias = Biased ? static_cast<uint_t>(params_t::IMPLICIT1) : uint_t{0};
    const uint_t d = static_cast<uint_t>(exp) << params_t::M;
    const uint_t yb = static_cast<uint_t>(x.to_bits() + bias - d);

    // round on the binary point
    const T t = round_to_integral<rm>(bit_float<T>(yb).to_float());

    // `t` is integral, so it differs from `y` exactly when digits were discarded -
    // the bias shifts both by the same integer, so it survives. The comparison is
    // bitwise, on encodings `t` needs anyway: every exact case in
    // `round_to_integral` reproduces `y` bit for bit, and no zero or NaN gets here.
    const uint_t tb = bit_float<T>(t).to_bits();
    const bool inexact = tb != yb;

    // scale back and remove the bias; `|t| >= 1`, so no sign is lost on the way
    return {bit_float<T>(static_cast<uint_t>(tb + d - bias)), inexact};
}

/// @brief The subnormal path of `round`: `x` is a non-zero subnormal.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
/// @return the rounded value
template <RM rm, flag_mask_t FlagMask, std::floating_point T>
bit_float<T> round_subnormal(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;

    // which flags to check
    static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_BEFORE = FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");
    MPFX_DEBUG_ASSERT(x.ebits() == 0, "normal, Inf, and NaN x are handled by round_scaled itself");

    // `x` is subnormal: compute the actual split point `n`
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
    // which the scaling relies on.
    if (n_act < params_t::EXPMIN) {
        // exact, so tininess after rounding matches tininess before
        if constexpr (CHECK_TINY_AFTER) {
            if (tiny_before) {
                flags.set_tiny_after_rounding();
            }
        }
        return x;
    }

    RoundResult<T> r;
    if (n_act >= e) {
        // `x` is nonzero and every digit was discarded, so this is always inexact
        r = {round_all_lost<rm>(x, e, n_act), true};
    } else {
        // only subnormal `x` reaches here, so the split is always biased
        r = round_split<rm, true>(x, n_act);
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

        round_internal::set_tiny_after<rm, FlagMask>(x, e, n_min, emin, r.inexact);
    } else if constexpr (CHECK_CARRY) {
        // see `round_general`; `x` is subnormal here, so this is exactly the case
        // a zero-mantissa-field test would miss
        const auto c = r.value.c();
        if ((c & (c - 1)) == 0 && r.value.compare_mag(x) > 0) {
            flags.set_carry();
        }
    }

    return r.value;
}

/// @brief The tiny path of `round`: a normal `x` whose split point is
/// subnormalized, `e - p < n`.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round, normal and tiny before rounding
/// @param p the target precision to round to
/// @param n the minimum normalized exponent for subnormalization
/// @param e the normalized exponent of `x`, which the caller already holds
/// @return the rounded value
template <RM rm, flag_mask_t FlagMask, std::floating_point T>
bit_float<T> round_tiny(bit_float<T> x, prec_t p, exp_t n, exp_t e) {
    using params_t = typename bit_float<T>::params_t;

    // which flags to check
    static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
    static constexpr bool CHECK_UNDERFLOW_BEFORE = FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");
    MPFX_DEBUG_ASSERT(!x.is_nar() && x.ebits() != 0, "x must be normal");
    MPFX_DEBUG_ASSERT(e == x.e(), "`e` must be the normalized exponent of `x`");
    MPFX_DEBUG_ASSERT(e - static_cast<exp_t>(p) < n, "the split must be subnormalized");

    // tiny before rounding, by this path's precondition
    if constexpr (CHECK_TINY_BEFORE) {
        flags.set_tiny_before_rounding();
    }

    // fast path: the split point is below every digit `x` can hold, so `x` is
    // already representable. This also keeps `n + 1` at or above `EXPMIN`,
    // which the scaling relies on.
    if (n < params_t::EXPMIN) {
        // exact, so tiny after rounding just as before it
        if constexpr (CHECK_TINY_AFTER) {
            flags.set_tiny_after_rounding();
        }
        return x;
    }

    RoundResult<T> r;
    if (n >= e) {
        // `x` is nonzero and every digit was discarded, so this is always inexact
        r = {round_all_lost<rm>(x, e, n), true};
    } else {
        // `x` is normal, so the split needs no bias
        r = round_split<rm, false>(x, n);
    }

    // set inexact flag if requested
    if constexpr (CHECK_INEXACT) {
        if (r.inexact) {
            flags.set_inexact();
        }
    }

    // underflow is tininess together with inexactness
    if constexpr (CHECK_UNDERFLOW_BEFORE) {
        if (r.inexact) {
            flags.set_underflow_before_rounding();
        }
    }

    const exp_t n_min = e - static_cast<exp_t>(p);
    const exp_t emin = n + static_cast<exp_t>(p);
    round_internal::set_tiny_after<rm, FlagMask>(x, e, n_min, emin, r.inexact);

    // no carry check: the carry flag is only defined for results that are not tiny
    return r.value;
}

/// @brief The normal path of `round`: a finite normal `x` whose split
/// point is not subnormalized, `n_min = e - p`.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round, finite, normal, and not tiny
/// @param p the target precision to round to
/// @param ebits the exponent field of `x`, which the caller already holds
/// @return the rounded value
template <RM rm, flag_mask_t FlagMask, std::floating_point T>
inline bit_float<T> round_normal(bit_float<T> x, prec_t p, typename bit_float<T>::uint_t ebits) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr bool CHECK_INEXACT = FlagMask & Flags::INEXACT_FLAG;
    static constexpr bool CHECK_CARRY = FlagMask & Flags::CARRY_FLAG;
    MPFX_DEBUG_ASSERT(x.ebits() == ebits && ebits != 0 && !x.is_nar(), "x must be normal");
    MPFX_DEBUG_ASSERT(p >= 1, "target precision must be at least one digit");
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");

    // fast path: the split point is below every digit `x` can hold, so `x` is
    // already representable. Exact and not tiny, so no flag can raise.
    const int64_t eb = static_cast<int64_t>(ebits >> params_t::M);
    if (eb < params_t::EXPMIN + static_cast<int64_t>(p) + params_t::BIAS) {
        return x;
    }

    // scale down: `2^-(n_min + 1)` sends every `x` of this shape into `|y|` in
    // `[2^(p-1), 2^p)`, so the scaled exponent field is a constant and the whole
    // factor is `ebits` minus it - no need for `e`
    const uint_t y_field = static_cast<uint_t>(params_t::BIAS + p - 1) << params_t::M;
    const uint_t scale = static_cast<uint_t>(ebits - y_field);
    const uint_t yb = static_cast<uint_t>(x.to_bits() - scale);

    // round on the binary point
    const T t = round_to_integral<rm>(bit_float<T>(yb).to_float());
    const uint_t tb = bit_float<T>(t).to_bits();

    // scale back
    const bit_float<T> r(static_cast<uint_t>(tb + scale));

    // set inexact flag if requested; exactness is bitwise, see `round_split`
    if constexpr (CHECK_INEXACT) {
        if (tb != yb) {
            flags.set_inexact();
        }
    }

    // Not tiny and `x` normal, so as in `round` the exponent field decides, the
    // saturation onto Inf included. Rounding never leaves the binade downward:
    // `|y| >= 2^(p-1)` keeps every mode's result at or above it.
    if constexpr (CHECK_CARRY) {
        if ((r.to_bits() & params_t::EMASK) > ebits) {
            flags.set_carry();
        }
    }

    return r;
}

/// @brief Rounding of a `bit_float` type by scaling - the implementation the public
/// `round` uses. Works in the floating-point rather than the integer domain, and is
/// independent of the host rounding mode. See `reference::round` for the oracle it
/// is checked against.
/// @tparam rm the rounding mode
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
/// @return the rounded value
///
/// Dispatches the shapes in order - Inf/NaN, subnormal `x`, a subnormalized split
/// of a normal `x`, then everything else - each to its own handler. The dispatch is
/// for the inliner as much as the reader: rounding is a few instructions per call,
/// so the common path dissolving into the caller's loop is worth more than any
/// instruction in it, and only bodies this small dissolve. The rarer handlers stay
/// real calls, so their cost falls only on the rare shapes.
template <RM rm, flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
bit_float<T> round(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    MPFX_DEBUG_ASSERT(p >= 1, "target precision must be at least one digit");
    MPFX_DEBUG_ASSERT(p <= params_t::P, "target precision cannot exceed the precision of the container type");
    MPFX_DEBUG_ASSERT(!n.has_value() || *n + 1 >= params_t::EXPMIN, "subnormalization point must be at least EMIN - 1");

    // extract exponent field
    const uint_t ebits = x.ebits();

    // Inf and NaN round to themselves
    if (ebits == params_t::EMASK) {
        return x;
    }

    // subnormal `x`, and zero, which shares its exponent field
    if (ebits == 0) {
        // which flags to check
        static constexpr bool CHECK_TINY_BEFORE = FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG;
        static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;

        // zero
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

        return round_subnormal<rm, FlagMask>(x, p, n);
    }

    // A subnormalized split of a normal `x`: subnormalization wins the `max` exactly
    // when `n_min < *n`, which is also exactly when `x` is tiny before rounding, so
    // this one comparison rules out every underflow concern at once.
    const int64_t eb = static_cast<int64_t>(ebits >> params_t::M);
    if (n.has_value() && eb < *n + static_cast<int64_t>(p) + params_t::BIAS) {
        return round_tiny<rm, FlagMask>(x, p, *n, static_cast<exp_t>(eb) - params_t::BIAS);
    }

    // everything else
    return round_normal<rm, FlagMask>(x, p, ebits);
}

/// @brief Rounding of a `bit_float` type by scaling, dispatching on a runtime
/// rounding mode. See the compile-time overload above.
/// @tparam FlagMask the mask of flags to set
/// @tparam T the floating-point container type
/// @param x the `bit_float` value to round
/// @param p the target precision to round to
/// @param n optional minimum normalized exponent for subnormalization
/// @param rm the rounding mode
/// @return the rounded value
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
        MPFX_DEBUG_ASSERT(false, "round_scaled: invalid rounding mode");
        return x; // default return to avoid warnings
    }
}


} // namespace round_scaled


/// The original rounding implementation, and the ancestor of the two above: it
/// decodes into a `(sign, exponent, significand)` triple and rounds with integer
/// arithmetic on the significand. Slowest of the three, and still what the
/// `round(m, exp, ...)` overload uses. Kept as a third, independent witness - it
/// shares no code with either of the others.
namespace round_reference {

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

} // namespace round_reference

/// @brief Rounds a floating-point number of type `T`
/// to a value of the same type with target precision `p` and first
/// unrepresentable digit `n`. Rounding happens in `T`'s own container, so a
/// `float` argument is not widened to `double` (which would double-round).
template<flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
T round(T x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    return round_scaled::round<FlagMask>(bit_float<T>(x), p, n, rm).to_float();
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
    return round_reference::round_finalize<PREC, U, FlagMask>(s, e, c, p, n, rm);
}

} // namespace mpfx
