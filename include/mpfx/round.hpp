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

/// @brief Is this value odd (i.e. has the least significant bit set)?
/// @tparam RM the rounding mode to use
/// @param x the value to check
/// @return true if x is odd, false otherwise
template <std::floating_point T>
bool is_odd(T x) {
    using uint_t = typename float_params<T>::uint_t;
    MPFX_DEBUG_ASSERT(std::isfinite(x), "is_odd is only defined for finite values");
    const uint_t bits = std::bit_cast<uint_t>(x);
    return bits & 1;
}

/// @brief Steps `x` by one ULP away from zero (increases magnitude).
/// Requires `x` to be finite. Works for `x == ±0` (yields ±smallest subnormal).
inline float next_away_zero(float x) {
    MPFX_DEBUG_ASSERT(std::isfinite(x), "next_away_zero requires a finite value");
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) + 1u);
}

/// @brief Steps `x` by one ULP toward zero (decreases magnitude).
/// Requires `x` to be finite and non-zero.
inline float next_toward_zero(float x) {
    MPFX_DEBUG_ASSERT(std::isfinite(x) && x != 0.0f, "next_toward_zero requires a finite non-zero value");
    return std::bit_cast<float>(std::bit_cast<uint32_t>(x) - 1u);
}

/// @brief Optimized rounding from `double` to `float`.
/// @tparam RM the rounding mode to use
/// @param x the `double` value to round to `float`
/// @return the rounded `float` value
///
/// Across all modes the implementation strategy is the same:
///   1. Let `y_rne = (float) x` — this is the hardware RNE result, which also
///      handles inf/NaN/+/-0 sign correctly.
///   2. Decide whether `y_rne` already matches the requested rounding mode; if
///      not, step it by one ULP in the bit pattern (away from zero = bits + 1,
///      toward zero = bits - 1) to produce the correct neighbor.
/// Each mode's comment gives the textbook pseudocode (treating x as a real
/// number with adjacent representable floats y_down <= x <= y_up) followed by
/// the y_rne-based shortcut that this code actually uses.
template <RM rm>
float fp64_to_fp32(double x) {
    static constexpr float POS_MAX = std::numeric_limits<float>::max();
    static constexpr double INFVAL = 3.4028236692093846e+38; // 2^128

    // round under RNE
    const float y_rne = static_cast<float>(x);

    if constexpr (rm == RM::RNE) {
        // round to nearest, ties to even: nothing to do
        return y_rne;
    } else if constexpr (rm == RM::RNA) {
        // round to nearest, ties away from zero): only disagrees at midpoints
        if (std::fabs(static_cast<double>(y_rne)) < std::fabs(x)) {
            // rounded down to zero so might be incorrect
            const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
            const float y_away = std::bit_cast<float>(bits + 1u);
            const double mid = (static_cast<double>(y_rne) + static_cast<double>(y_away)) * 0.5;
            return x == mid ? y_away : y_rne;
        } else {
            // definitely correct
            return y_rne;
        }
    } else if constexpr (rm == RM::RTZ) {
        // round toward zero: check if we rounded away from zero and if so
        // step back toward zero by one ULP
        const bool down = std::fabs(static_cast<double>(y_rne)) > std::fabs(x);
        const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
        return std::bit_cast<float>(bits - static_cast<uint32_t>(down));
    } else if constexpr (rm == RM::RAZ) {
        // round away from zero: check if we rounded toward zero and if so
        // step away from zero by one ULP.
        const bool up = std::fabs(static_cast<double>(y_rne)) < std::fabs(x);
        const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
        return std::bit_cast<float>(bits + static_cast<uint32_t>(up));
    } else if constexpr (rm == RM::RTP) {
        if (y_rne < x) {
            const uint32_t delta = std::signbit(y_rne) ? static_cast<uint32_t>(-1) : 1u;
            const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
            return std::bit_cast<float>(bits + delta);
        } else {
            return y_rne;
        }
    } else if constexpr (rm == RM::RTN) {
        // round toward -infinity: check if we rounded up and if so step down by one ULP.
        if (y_rne > x) {
            const uint32_t delta = std::signbit(y_rne) ? 1u : static_cast<uint32_t>(-1);
            const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
            return std::bit_cast<float>(bits + delta);
        } else {
            return y_rne;
        }
    } else if constexpr (rm == RM::RTO) {
        // round to odd: if already odd or exact, keep; otherwise step one ULP away
        // from zero (bits + 1 for positive, bits - 1 for negative)
        if (!std::isfinite(y_rne)) {
            if (std::isnan(x)) {
                return y_rne;
            }
            return (std::abs(x) >= INFVAL) ? y_rne : std::copysign(POS_MAX, x);
        }
        const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
        if ((bits & 1u) || static_cast<double>(y_rne) == x) {
            // either exact or odd, so already correct
            return y_rne;
        }
        // step toward x by one ULP: away-from-zero iff (y_rne < x) xor signbit
        const bool up_in_value = static_cast<double>(y_rne) < x;
        const bool away = up_in_value ^ (bits >> 31);
        const uint32_t delta = away ? 1u : static_cast<uint32_t>(-1);
        return std::bit_cast<float>(bits + delta);
    } else if constexpr (rm == RM::RTE) {
        // round to even: if already even or exact, keep; otherwise step one ULP toward
        // the even neighbor
        if (!std::isfinite(y_rne)) {
            return y_rne;
        }
        const uint32_t bits = std::bit_cast<uint32_t>(y_rne);
        if (!(bits & 1u) || static_cast<double>(y_rne) == x) {
            // either exact or even, so already correct
            return y_rne;
        }
        const bool up_in_value = static_cast<double>(y_rne) < x;
        const bool away = up_in_value ^ (bits >> 31);
        const uint32_t delta = away ? 1u : static_cast<uint32_t>(-1);
        return std::bit_cast<float>(bits + delta);
    } else {
        MPFX_DEBUG_ASSERT(false, "fp64_to_fp32: unsupported rounding mode");
        return static_cast<float>(NAN);
    }
}

/// @brief Optimized rounding from `double` to `float`.
/// @param x the `double` value to round to `float`
/// @param rm the rounding mode to use
/// @return the rounded `float` value
inline float fp64_to_fp32(double x, RM rm) {
    switch (rm) {
        case RM::RNE:
            return fp64_to_fp32<RM::RNE>(x);
        case RM::RNA:
            return fp64_to_fp32<RM::RNA>(x);
        case RM::RTP:
            return fp64_to_fp32<RM::RTP>(x);
        case RM::RTN:
            return fp64_to_fp32<RM::RTN>(x);
        case RM::RTZ:
            return fp64_to_fp32<RM::RTZ>(x);
        case RM::RAZ:
            return fp64_to_fp32<RM::RAZ>(x);
        case RM::RTO:
            return fp64_to_fp32<RM::RTO>(x);
        case RM::RTE:
            return fp64_to_fp32<RM::RTE>(x);
        default:
            MPFX_DEBUG_ASSERT(false, "fp64_to_fp32: unsupported rounding mode");
            return static_cast<float>(x); // default return to avoid warnings
    }
}

/// @brief Should we increment to round?
/// @tparam RM the rounding mode
/// @tparam T the type of the significand
/// @param hi the high part of the split significand
/// @param n the split point
/// @param halfway whether the low part is exactly at the halfway point
/// @param sticky whether the low part has any nonzero bits below the halfway point
/// @return should we increment the significand?
template <RM rm, std::floating_point T>
inline bool round_increment_nearest(bit_float<T> hi, exp_t n, bool halfway, bool sticky) {
    // case split on rounding mode
    if constexpr (rm == RM::RNE) {
        if (halfway && !sticky) {
            // exactly halfway - increment if the LSB is odd
            return hi.bit(n + 1);
        } else {
            // above halfway - increment
            return halfway;
        }
    } else if constexpr (rm == RM::RNA) {
        // above or exactly at halfway - increment
        return halfway;
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
        const auto [hi, halfway, sticky] = x.split_rs(n - 1);
        return !round_increment_nearest<rm>(hi, n - 1, halfway, sticky);
    } else {
        // directed rounding modes
        const auto [hi, sticky] = x.split_sticky(n - 1);
        if (!sticky) {
            // exactly representable
            return true;
        } else {
            // not exact incremental
            return !round_increment_directed<rm>(hi, n - 1);
        }
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
    MPFX_DEBUG_ASSERT(p < params_t::P, "target precision must be less than the precision of the container type");
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
        const auto [hi, halfway, sticky] = x.split_rs(n_act);

        // fast path: low is zero
        if (!halfway && !sticky) {
            // we are tiny after rounding if we were tiny before rounding
            if constexpr (CHECK_TINY_AFTER) {
                if (tiny_before) {
                    flags.set_tiny_after_rounding();
                }
            }

            return hi;
        }

        // should we increment?
        increment = round_increment_nearest<rm>(hi, n_act, halfway, sticky);
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

    MPFX_DEBUG_ASSERT(p <= FP::P, "cannot keep the requested precision" << p);

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

/// @brief Optimized rounding to round a double-precision floating-point number
/// to a double-precision floating-point number with target precision `p`
/// and first unrepresentable digit `n`.
template<flag_mask_t FlagMask = Flags::ALL_FLAGS, std::floating_point T>
T round(T x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    return experimental::round<FlagMask>(bit_float<double>(x), p, n, rm).to_float();
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
