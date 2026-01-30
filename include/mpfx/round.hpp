#pragma once

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
/// floating-point number.
template <prec_t P>
double encode(bool s, exp_t e, mant_t c) {
    using FP = ieee754_consts<11, 64>; // double precision

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

/// @brief Finalizes the rounding procedure.
/// @tparam P the precision of the significand `c`
/// @param s sign
/// @param e normalized exponent
/// @param c integer significand
/// @param p precision to keep
/// @param rm rounding mode
/// @param overshiftp are we overshifting all digits?
/// @return the correctly rounded result as a `double`
template <prec_t P>
double round_finalize(bool s, exp_t e, mant_t c, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = ieee754_consts<11, 64>; // double precision
    MPFX_STATIC_ASSERT(P <= 63, "mantissa cannot be 64 bits");
    MPFX_DEBUG_ASSERT(p <= FP::P, "cannot keep the requested precision" << p);

    if (c == 0) {
        // fast path: zero value

        // raise tiny flags
        tiny_before_rounding_flag = true;
        tiny_after_rounding_flag = true;

        // return +/-0
        return s ? -0.0 : 0.0;
    }

    // our precision might be limited by subnormalization
    bool overshiftp = false; // are all digits insignificant and non-adjacent to n?
    bool tiny_before_rounding = false; // was the value tiny before rounding?
    exp_t emin = 0; // emin for tininess check (unused if n has no value)
    if (n.has_value()) {
        const exp_t nx = e - p;
        const exp_t offset = *n - nx;
        if (offset > 0) {
            // precision reduced due to subnormalization

            // compute the minimum normalized exponent
            emin = *n + p;
            tiny_before_rounding = e < emin;

            // set tiny before rounding flag
            if (tiny_before_rounding) {
                tiny_before_rounding_flag = true;
            }

            // "overshift" is set if we shift more than p bits
            const prec_t offset_pos = static_cast<prec_t>(offset);
            overshiftp = offset_pos > p; // set overshift flag
            p = overshiftp ? 0 : p - offset_pos; // precision cannot be negative
            e = overshiftp ? *n : e; // overshift implies e < n, set for correct increment to MIN_VAL
        }
    }

    // extract discarded bits
    const prec_t p_lost = p < P ? P - p : 0;
    const mant_t c_mask = bitmask<mant_t>(p_lost);
    const mant_t c_lost = c & c_mask;

    // clear discarded bits
    c &= ~c_mask;

    // check if we rounded off any significant digits
    if (c_lost != 0) {
        // slow path: inexact result
        MPFX_DEBUG_ASSERT(p_lost > 0, "we must have lost precision");

        // set inexact flag
        inexact_flag = true;

        // set underflow (before rounding) flag if tiny before rounding
        if (tiny_before_rounding_flag) {
            underflow_before_rounding_flag = true;
        }

        // value of the LSB for precision p
        const mant_t one = 1ULL << p_lost;

        // should we increment?
        // case split on nearest
        bool incrementp;
        switch (rm) {
            case RM::RNE: {
                // nearest rounding, ties to even

                // clever way to extract rounding information
                // -1: below halfway
                //  0: exactly halfway
                //  1: above halfway
                const mant_t halfway = one >> 1;
                const int8_t cmp = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);
                const int8_t rb = overshiftp ? -1 : cmp; // overshift implies below halfway

                // increment if above halfway or exactly halfway and LSB is odd
                incrementp = rb > 0 || (rb == 0 && (c & one) != 0);
                break;
            }
            case RM::RNA: {
                // nearest rounding, ties away from zero

                // clever way to extract rounding information
                // -1: below halfway
                //  0: exactly halfway
                //  1: above halfway
                const mant_t halfway = one >> 1;
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
                incrementp = (c & one) == 0;
                break;
            case RM::RTE:
                // round to even
                incrementp = (c & one) != 0;
                break;
            default:
                incrementp = false;
                MPFX_DEBUG_ASSERT(false, "unreachable");
                break;
        }

        if (incrementp) {
            // apply increment
            c += one;

            // check if we need to carry
            static constexpr mant_t overflow_mask = 1ULL << P;
            if (c >= overflow_mask) {
                // increment caused carry
                e += 1;
                c >>= 1;
                carry_flag = true;
            }
        }
    }

    return encode<P>(s, e, c);
}

} // anonymous namespace

/// @brief Optimized rounding to round a double-precision floating-point number
/// to a double-precision floating-point number with target precision `p`
/// and first unrepresentable digit `n`.
///
/// Assumes that the argument has at least p + 2 bits of precision,
/// where p is the target precision.
inline double round(double x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = ieee754_consts<11, 64>; // double precision

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
        // subnormal
        e = FP::EMIN;
        c = mbits;
        if (c != 0) {
            // non-zero => fully normalize the significand
            const auto lz = FP::P - std::bit_width(mbits);
            e -= static_cast<exp_t>(lz);
            c <<= lz;
        }
    } else {
        // normal (assuming no infinity or NaN)
        e = static_cast<exp_t>(ebits) - FP::BIAS;
        c = FP::IMPLICIT1 | mbits;
    }

    // finalize rounding (mantissa has precision `FP::P`)
    return round_finalize<FP::P>(s, e, c, p, n, rm);
}

/// @brief Optimized rounding to round `m * 2^exp`
/// to a double-precision floating-point number with target precision `p`
/// and first unrepresentable digit `n`.
///
/// Assumes that the argument has at least p + 2 bits of precision,
/// where p is the target precision.
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
    return round_finalize<PREC>(s, e, c, p, n, rm);
}

} // namespace mpfx
