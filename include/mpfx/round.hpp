#pragma once

#include <cmath>
#include <optional>
#include <tuple>

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

    const size_t idx = (static_cast<size_t>(mode) << 1) | static_cast<size_t>(sign);
    return table[idx];
}

namespace {

/// @brief Encodes the result of rounding as a double-precision
/// floating-point number. This is an optimized version of `make_float<double>`
/// which assumes that `c` is either 0 or has precision exactly `P`.
template <prec_t P>
double encode(bool s, exp_t e, mant_t c) {
    using FP = float_params<double>::params; // double precision

    // for encoding we need to ensure that we have 53 bits of precision
    // we cannot lose bits since we guarded against too much precision,
    // i.e., `c` has at most 63 bits of precision
    if constexpr (P > FP::P) {
        // `c` has more than 53 bits of precision
        static constexpr prec_t shift_p = P - FP::P;
        static constexpr mant_t excess_mask = __bitmask<mant_t, shift_p>::val;
        MPFX_DEBUG_ASSERT((c & excess_mask) == 0, "shifting off digits");
        c >>= shift_p;
    } else if constexpr (P < FP::P) {
        // `c` has less than 53 bits of precision
        static constexpr prec_t shift_p = FP::P - P;
        c <<= shift_p;
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
        mbits = c >> shift;
    } else {
        // normal result - most common case
        ebits = e + FP::BIAS;
        mbits = c & FP::MMASK;
    }

    // repack the result
    const uint64_t b = (ebits << FP::M) | mbits;
    const double r = std::bit_cast<double>(b);
    return s ? -r : r;
}

/// @brief Should we increment to round?
/// @param s sign
/// @param c_kept current significand
/// @param c_lost lost significand bits
/// @param p_lost number of lost precision bits
/// @param overshiftp are we overshifting all digits?
/// @param rm rounding mode
/// @return should we increment the significand?
inline bool round_increment(bool s, mant_t c_kept, mant_t c_lost, prec_t p_lost, bool overshiftp, RM rm) {
    MPFX_DEBUG_ASSERT(p_lost > 0, "we must have lost precision");

    // case split on rounding mode
    switch (rm) {
        case RM::RNE:
        case RM::RNA: {
            // nearest rounding modes - factor out common logic
            // Compute rounding bit: -1 (below halfway), 0 (exactly halfway), 1 (above halfway)
            const mant_t halfway = 1ULL << (p_lost - 1);
            const int8_t cmp = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);
            const int8_t rb = overshiftp ? -1 : cmp; // overshift implies below halfway

            if (rb > 0) [[likely]] {
                // above halfway - always increment
                return true;
            } else if (rb < 0) [[likely]] {
                // below halfway - never increment
                return false;
            } else [[unlikely]] {
                // exactly at halfway - tie-breaking (rare)
                if (rm == RM::RNE) {
                    // ties to even: increment if LSB is odd
                    return (c_kept >> p_lost) & 0x1;
                } else {
                    // ties away from zero: always increment
                    return true;
                }
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
/// @param rm rounding mode
/// @param overshiftp are we overshifting all digits?
/// @return the correctly rounded result as a `double`
template <prec_t P, flag_mask_t FlagMask = Flags::ALL_FLAGS>
double round_finalize(bool s, exp_t e, mant_t c, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = float_params<double>::params; // double precision
    static constexpr size_t MAX_C_WIDTH = 8 * sizeof(mant_t) - 1; // -1 to tolerate a carry
    static constexpr exp_t MAX_E = FP::EMAX + 1;
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
    exp_t emin = MAX_E;       // minimum normalized exponent (if n is set)
    bool overshiftp = false;  // are all digits insignificant and non-adjacent to n?
    bool tiny_before = false; // was the value tiny before rounding?
    bool tiny_after = false;  // was the value tiny after rounding?

    // handle possible subnormalization
    if (n.has_value()) {
        // compute the minimum normalized exponent
        emin = *n + static_cast<exp_t>(p);
        tiny_before = e < emin;

        if (tiny_before) {
            // our precision is limited by subnormalization

            // set tiny before rounding flag
            if constexpr (CHECK_TINY_BEFORE) {
                flags.set_tiny_before_rounding();
            }

            // check for tininess after rounding
            if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
                // check for the easy case of tininess after rounding
                if (e < emin - 1) {
                    // definitely tiny after rounding, since we are at least
                    // one binade below the smallest normal number
                    tiny_after = true;
                } else {
                    // we are in the largest binade below 2^emin
                    // significand of the largest representable value below 2^emin
                    // the cutoff value is always odd: 1.111...111 x 2^(emin-1)
                    const mant_t cutoff = bitmask<mant_t>(p) << (P - p);
                    if (c <= cutoff) {
                        tiny_after = true;
                    } else {
                        // otherwise, we are in the hard case and need to check
                        // for tininess after splitting the significand
                    }
                }
            }

            // "overshift" is set if we shift more than p bits
            const prec_t shift = static_cast<prec_t>(emin - e);
            overshiftp = shift > p; // set overshift flag
            p_kept = overshiftp ? 0 : p - shift; // precision cannot be non-positive
            e = overshiftp ? *n : e; // set exponent for subnormalization
        }
    }

    // split bits into kept and lost parts
    const prec_t p_lost = p_kept < P ? P - p_kept : 0;
    const mant_t c_mask = bitmask<mant_t>(p_lost);
    const mant_t c_lost = c & c_mask;
    mant_t c_kept = c & ~c_mask;

    // check if we rounded off any significant digits
    if (c_lost != 0) {
        // slow path: inexact result
        MPFX_DEBUG_ASSERT(p_lost > 0, "we must have lost precision");

        // check the hard case for tiny after rounding
        if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
            if (tiny_before && !tiny_after) [[unlikely]] {
                // we are just below 2^emin but above the cutoff value
                MPFX_DEBUG_ASSERT(n.has_value(), "n must be set");
                MPFX_DEBUG_ASSERT(p_kept < P, "must have kept at least one digit");
                MPFX_DEBUG_ASSERT(p_lost > 1, "must have lost at least 2 digits");
                MPFX_DEBUG_ASSERT(e == emin - 1, "must be in the largest binade below 2^emin");
                MPFX_DEBUG_ASSERT(!overshiftp, "must not have overshifted all digits");

                // need to check if we round to 2^emin (unbounded exponent)
                // by rounding with a split that is one digit lower
                const mant_t one = 1ULL << (p_lost - 1); // dummy value to indicate oddness
                const mant_t c_half_mask = bitmask<mant_t>(p_lost - 1);
                const mant_t c_lost_half = c_lost & c_half_mask;
                tiny_after = !round_increment(s, one, c_lost_half, p_lost - 1, false, rm);
            }

            // set tiny after rounding flag
            if constexpr (CHECK_TINY_AFTER) {
                if (tiny_after) {
                    flags.set_tiny_after_rounding();
                }
            }
        }

        // should we increment?
        if (round_increment(s, c_kept, c_lost, p_lost, overshiftp, rm)) {
            // size of the increment
            const mant_t one = 1ULL << p_lost;

            // apply increment
            c_kept += one;

            // check if we need to carry
            static constexpr mant_t overflow_mask = 1ULL << P;
            if (c_kept >= overflow_mask) [[unlikely]] {
                // increment caused carry
                e += 1;
                c_kept >>= 1;
                if constexpr (CHECK_CARRY) {
                    if (e > emin) {
                        flags.set_carry();
                    }
                }
            }
        }

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
    } else {
        // exact result

        // set tiny after rounding flag if tiny before
        if constexpr (CHECK_TINY_AFTER) {
            if (tiny_before) {
                flags.set_tiny_after_rounding();
            }
        }
    }

    return encode<P>(s, e, c_kept);
}

} // anonymous namespace

/// @brief Optimized rounding to round a double-precision floating-point number
/// to a double-precision floating-point number with target precision `p`
/// and first unrepresentable digit `n`.
///
/// Assumes that the argument has at least p + 2 bits of precision,
/// where p is the target precision.
template<flag_mask_t FlagMask = Flags::ALL_FLAGS>
double round(double x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = float_params<double>::params; // double precision

    // Fast path: special values (infinity, NaN)
    if (!std::isfinite(x)) {
        return x;
    }

    // decode floating-point data
    auto [s, exp, c] = unpack_float<double>(x);

    // fully normalize the significand
    const prec_t xp = std::bit_width(c);
    if (xp < FP::P) [[unlikely]] {
        // subnormal input
        const prec_t lz = FP::P - xp;
        c <<= lz;
        exp -= static_cast<exp_t>(lz);
    }

    // compute normalized exponent
    const exp_t e = exp + static_cast<exp_t>(FP::P - 1);

    // finalize rounding (mantissa has precision `FP::P`)
    return round_finalize<FP::P, FlagMask>(s, e, c, p, n, rm);
}

/// @brief Optimized rounding to round `m * 2^exp`
/// to a double-precision floating-point number with target precision `p`
/// and first unrepresentable digit `n`.
///
/// Assumes that the argument has at least p + 2 bits of precision,
/// where p is the target precision.
template<flag_mask_t FlagMask = Flags::ALL_FLAGS>
double round(int64_t m, exp_t exp, prec_t p, const std::optional<exp_t>& n, RM rm) {
    static constexpr int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
    static constexpr prec_t PREC = 63;

    // Decode `m` into sign-magnitude
    bool s;
    mant_t c;
    if (m < 0) {
        s = true;
        if (m == MIN_VAL) {
            // special decode to ensure 63 bits of precision
            c = 1ULL << (PREC - 1);
            exp += 1;
        } else {
            // normal decode
            c = static_cast<mant_t>(std::abs(m));
        }
    } else {
        s = false;
        c = static_cast<mant_t>(m);
    }

    // we may have less precision than expected
    // guaranteed to have at most 63 bits
    const auto lz = PREC - std::bit_width(c);
    c <<= lz;
    exp -= lz;

    // calculate normalized exponent
    const exp_t e = exp + (PREC - 1);

    // finalize rounding (mantissa has precision 63)
    return round_finalize<PREC, FlagMask>(s, e, c, p, n, rm);
}

} // namespace mpfx
