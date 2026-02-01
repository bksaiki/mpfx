#pragma once

#include <cmath>
#include <optional>
#include <tuple>

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
    switch (mode) {
        case RoundingMode::RNE:
            return RoundingDirection::TO_EVEN;
        case RoundingMode::RNA:
            return RoundingDirection::AWAY_ZERO;
        case RoundingMode::RTP:
            return sign ? RoundingDirection::TO_ZERO : RoundingDirection::AWAY_ZERO;
        case RoundingMode::RTN:
            return sign ? RoundingDirection::AWAY_ZERO : RoundingDirection::TO_ZERO;
        case RoundingMode::RTZ:
            return RoundingDirection::TO_ZERO;
        case RoundingMode::RAZ:
            return RoundingDirection::AWAY_ZERO;
        case RoundingMode::RTO:
            return RoundingDirection::TO_ODD;
        case RoundingMode::RTE:
            return RoundingDirection::TO_EVEN;
        default:
            MPFX_UNREACHABLE("invalid rounding mode");
    }
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
    if (UNLIKELY(c == 0)) {
        // edge case: subnormalization underflowed to 0
        // `e` might be an unexpected value here
        ebits = 0;
        mbits = 0;
    } else if (UNLIKELY(e < FP::EMIN)) {
        // subnormal result
        const exp_t shift = FP::EMIN - e;
        ebits = 0;
        mbits = c >> shift;
    } else {
        // normal result
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
/// @param c current significand
/// @param c_lost lost significand bits
/// @param p_lost number of lost precision bits
/// @param overshiftp are we overshifting all digits?
/// @param rm rounding mode
/// @return should we increment the significand?
inline bool round_increment(
    bool s,
    bool odd,
    mant_t c_lost,
    prec_t p_lost,
    bool overshiftp,
    RM rm
) {
    MPFX_DEBUG_ASSERT(p_lost > 0, "we must have lost precision");

    // case split on rounding mode
    bool incrementp;
    switch (rm) {
        case RM::RNE: {
            // nearest rounding, ties to even

            // clever way to extract rounding information
            // -1: below halfway
            //  0: exactly halfway
            //  1: above halfway
            const mant_t halfway = 1ULL << (p_lost - 1);
            const int8_t cmp = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);
            const int8_t rb = overshiftp ? -1 : cmp; // overshift implies below halfway

            // increment if above halfway or exactly halfway and LSB is odd
            incrementp = rb > 0 || (rb == 0 && odd);
            break;
        }
        case RM::RNA: {
            // nearest rounding, ties away from zero

            // clever way to extract rounding information
            // -1: below halfway
            //  0: exactly halfway
            //  1: above halfway
            const mant_t halfway = 1ULL << (p_lost - 1);
            const int8_t cmp = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);
            const int8_t rb = overshiftp ? -1 : cmp; // overshift implies below halfway

            // increment if above halfway or exactly halfway
            incrementp = rb >= 0;
            break;
        }
        case RM::RTP:
            // round toward +infinity
            incrementp = !s;
            break;
        case RM::RTN:
            // round toward -infinity
            incrementp = s;
            break;
        case RM::RTZ:
            // round toward zero
            incrementp = false;
            break;
        case RM::RAZ:
            // round away from zero
            incrementp = true;
            break;
        case RM::RTO:
            // round to odd
            incrementp = !odd;
            break;
        case RM::RTE:
            // round to even
            incrementp = !odd;
            break;
        default:
            incrementp = false;
            MPFX_DEBUG_ASSERT(false, "unreachable");
            break;
    }

    return incrementp;
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
    MPFX_STATIC_ASSERT(P <= 63, "mantissa cannot be 64 bits");
    MPFX_DEBUG_ASSERT(p <= FP::P, "cannot keep the requested precision" << p);
    static constexpr exp_t MAX_E = FP::EMAX + 1;

    if (c == 0) {
        // fast path: zero value
        // raise both tiny flags
        if constexpr (FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG) {
            flags.set_tiny_before_rounding();
        }
        if constexpr (FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG) {
            flags.set_tiny_after_rounding();
        }

        // return +/-0
        return s ? -0.0 : 0.0;
    }

    exp_t e_before = e;       // exponent before rounding
    prec_t p_kept = p;        // actual precision kept
    exp_t emin = MAX_E;       // minimum normalized exponent (if n is set)
    bool overshiftp = false;  // are all digits insignificant and non-adjacent to n?
    bool tiny_before = false; // was the value tiny before rounding?

    // our precision might be limited by subnormalization
    if (n.has_value()) {
        // n is set => we may need to subnormalize
        emin = *n + static_cast<exp_t>(p);
        tiny_before = e < emin;

        if (tiny_before) {
            // set tiny before rounding flag
            if constexpr (FlagMask & Flags::TINY_BEFORE_ROUNDING_FLAG) {
                flags.set_tiny_before_rounding();
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

        // set inexact flag
        if constexpr (FlagMask & Flags::INEXACT_FLAG) {
            flags.set_inexact();
        }

        // if subnormal before rounding, multiple things to check
        if (tiny_before) {
            // tininess before => we should raise underflow before rounding flag
            // and check for tiny after rounding
            MPFX_DEBUG_ASSERT(n.has_value(), "n must be set");

            // set the underflow before rounding flag
            if constexpr (FlagMask & Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG) {
                flags.set_underflow_before_rounding();
            }

            static constexpr bool CHECK_TINY_AFTER = FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG;
            static constexpr bool CHECK_UNDERFLOW_AFTER = FlagMask & Flags::UNDERFLOW_AFTER_ROUNDING_FLAG;

            // check if we are tiny after rounding
            if constexpr (CHECK_TINY_AFTER || CHECK_UNDERFLOW_AFTER) {
                bool tiny_after;
                if (e_before < emin - 1) {
                    // definitely tiny after rounding, since we are at least
                    // one binade below the smallest normal number
                    tiny_after = true;
                } else {
                    // possibly not tiny: we are in the largest binade below 2^emin
                    MPFX_DEBUG_ASSERT(p_kept < P, "must have kept at least one digit");
                    MPFX_DEBUG_ASSERT(p_lost > 1, "must have lost at least 2 digits");
                    MPFX_DEBUG_ASSERT(e == emin - 1, "must be in the largest binade below 2^emin");
                    MPFX_DEBUG_ASSERT(!overshiftp, "must not have overshifted all digits");

                    // significand of the largest representable value below 2^emin
                    // the cutoff value is always odd: 1.111...111 x 2^(emin-1)
                    const mant_t cutoff = bitmask<mant_t>(p) << (P - p);

                    if (c <= cutoff) {
                        // definitely tiny: we are smaller than or equal to the
                        // largest representable value below 2^emin (unbounded exponent)
                        tiny_after = true;
                    } else {
                        // hard case: we are larger than the cutoff value
                        // need to check if we round to 2^emin (unbounded exponent)
                        // by rounding with a split that is one digit lower
                        const mant_t c_half_mask = bitmask<mant_t>(p_lost - 1);
                        const mant_t c_lost_half = c_lost & c_half_mask;
                        tiny_after = !round_increment(s, true, c_lost_half, p_lost - 1, false, rm);
                    }
                }

                // set tiny after rounding flag
                if (tiny_after) {
                    if constexpr (FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG) {
                        flags.set_tiny_after_rounding();
                    }
                    if constexpr (FlagMask & Flags::UNDERFLOW_AFTER_ROUNDING_FLAG) {
                        flags.set_underflow_after_rounding();
                    }
                }
            }
        }

        // size of the increment
        const mant_t one = 1ULL << p_lost;

        // is the mantissa odd?
        const bool odd = (c_kept & one) != 0;

        // should we increment?
        if (round_increment(s, odd, c_lost, p_lost, overshiftp, rm)) {
            // apply increment
            c_kept += one;

            // check if we need to carry
            static constexpr mant_t overflow_mask = 1ULL << P;
            if (c_kept >= overflow_mask) {
                // increment caused carry
                e += 1;
                c_kept >>= 1;
                if constexpr (FlagMask & Flags::CARRY_FLAG) {
                    if (e > emin) {
                        flags.set_carry();
                    }
                }
            }
        }
    } else {
        // exact result
        // set tiny after rounding flag if tiny before
        if (tiny_before) {
            if constexpr (FlagMask & Flags::TINY_AFTER_ROUNDING_FLAG) {
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
inline double round(double x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = float_params<double>::params; // double precision

    // Fast path: special values (infinity, NaN, zero)
    if (!std::isfinite(x)) {
        return x;
    }

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const bool s = (b >> (FP::N - 1)) != 0;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decode floating-point data
    exp_t e;
    mant_t c;
    if (UNLIKELY(ebits == 0)) {
        // subnormal => fully normalize the significand
        e = FP::EMIN;
        c = mbits;

        const auto lz = FP::P - std::bit_width(mbits);
        e -= static_cast<exp_t>(lz);
        c <<= lz;
    } else {
        // normal (assuming no infinity or NaN)
        e = static_cast<exp_t>(ebits) - FP::BIAS;
        c = FP::IMPLICIT1 | mbits;
    }

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
inline double round(int64_t m, exp_t exp, prec_t p, const std::optional<exp_t>& n, RM rm) {
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
