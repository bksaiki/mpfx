#pragma once

#include <bit>
#include <cmath>
#include <limits>

#include "context.hpp"
#include "params.hpp"
#include "types.hpp"

namespace mpfx {

/// @brief Describes how NaN values are encoded for an `EFloatContext`.
///
/// See https://uwplse.org/2025/02/17/Small-Floats.html for details.
enum class EFloatNanKind : uint8_t {
    /// @brief IEEE 754 compliant: NaNs have the largest exponent.
    IEEE_754,
    /// @brief NaN has the largest exponent and mantissa of all ones.
    MAX_VAL,
    /// @brief NaN replaces negative zero.
    NEG_ZERO,
    /// @brief No NaNs.
    NONE,
};

namespace {

/// @brief A finite value `c * 2^exp` with integer significand `c`.
struct efloat_binade {
    mant_t c;
    exp_t exp;
};

constexpr prec_t efloat_prec(prec_t es, prec_t nbits) {
    return nbits - es;
}

/// @brief The exponent bias `2^(es-1) - 1` (0 when `es == 0`).
constexpr exp_t efloat_ebias(prec_t es) {
    return es == 0 ? 0 : static_cast<exp_t>(bitmask<uint32_t>(es - 1));
}

/// @brief The maximum normalized exponent, including the offset.
constexpr exp_t efloat_emax(prec_t es, exp_t eoffset) {
    const exp_t emax_0 = es == 0 ? -1 : efloat_ebias(es);
    return emax_0 + eoffset;
}

/// @brief The minimum normalized exponent, including the offset.
constexpr exp_t efloat_emin(prec_t es, exp_t eoffset) {
    return (1 - efloat_ebias(es)) + eoffset;
}

/// @brief The largest value in the binade with normalized exponent `e`,
/// i.e. `(2^p - 1) * 2^(e - p + 1)`, accounting for subnormalization at `emin`.
constexpr efloat_binade efloat_binade_max(prec_t p, exp_t emin, exp_t e) {
    if (e >= emin) {
        return { bitmask<mant_t>(p), e - static_cast<exp_t>(p) + 1 };
    }
    const exp_t shift = emin - e;
    return { bitmask<mant_t>(p) >> shift, emin - static_cast<exp_t>(p) + 1 };
}

/// @brief The next representable value towards zero, at precision `p` with
/// first unrepresentable digit `nmin` (mirrors `RealFloat.next_towards_zero`).
constexpr efloat_binade efloat_next_towards_zero(efloat_binade x, prec_t p, exp_t nmin) {
    mant_t c = x.c - 1;
    exp_t exp = x.exp;
    // if we crossed a power-of-two boundary, borrow a lower bit
    if (exp > nmin + 1 && static_cast<prec_t>(bit_width(c)) < p) {
        c = (c << 1) | 1;
        exp -= 1;
    }
    return { c, exp };
}

/// @brief Encodes a finite value `c * 2^exp` as a double. Mirrors
/// `make_float<double>`, but `constexpr` (relies only on `std::bit_cast`).
/// Loses precision if the significand exceeds 53 bits, consistent with
/// representing `maxval` as a double.
constexpr double efloat_binade_to_double(efloat_binade x) {
    using FP = float_params<double>::params;
    if (x.c == 0) {
        return 0.0;
    }

    // normalize the significand to `FP::P` bits
    mant_t c = x.c;
    const prec_t p = static_cast<prec_t>(bit_width(c));
    const exp_t e = x.exp + static_cast<exp_t>(p) - 1; // normalized exponent
    const exp_t shift = static_cast<exp_t>(FP::P) - static_cast<exp_t>(p);
    if (shift > 0) {
        c <<= shift;
    } else if (shift < 0) {
        c >>= -shift;
    }

    // encode the exponent and mantissa fields
    uint64_t ebits, mbits;
    if (e < FP::EMIN) {
        ebits = 0;
        mbits = static_cast<uint64_t>(c >> (FP::EMIN - e));
    } else {
        ebits = static_cast<uint64_t>(e + FP::BIAS);
        mbits = static_cast<uint64_t>(c) & FP::MMASK;
    }

    const uint64_t bits = (ebits << FP::M) | mbits;
    return std::bit_cast<double>(bits);
}

/// @brief The largest finite representable magnitude for an EFloat format,
/// as a double. Assumes `p >= 2`.
constexpr double efloat_max_value(
    prec_t es, prec_t nbits, bool enable_inf, EFloatNanKind nan_kind, exp_t eoffset)
{
    const prec_t p = efloat_prec(es, nbits);
    const exp_t emax = efloat_emax(es, eoffset);
    const exp_t emin = efloat_emin(es, eoffset);
    const exp_t nmin = emin - static_cast<exp_t>(p);

    efloat_binade bm;
    switch (nan_kind) {
        case EFloatNanKind::IEEE_754:
            bm = efloat_binade_max(p, emin, emax);
            break;
        case EFloatNanKind::MAX_VAL:
            if (p == 2 && enable_inf) {
                bm = efloat_binade_max(p, emin, emax);
            } else if (enable_inf) {
                bm = efloat_binade_max(p, emin, emax + 1);
                bm = efloat_next_towards_zero(bm, p, nmin);
                bm = efloat_next_towards_zero(bm, p, nmin);
            } else {
                bm = efloat_binade_max(p, emin, emax + 1);
                bm = efloat_next_towards_zero(bm, p, nmin);
            }
            break;
        case EFloatNanKind::NEG_ZERO:
        case EFloatNanKind::NONE:
            if (enable_inf) {
                bm = efloat_binade_max(p, emin, emax + 1);
                bm = efloat_next_towards_zero(bm, p, nmin);
            } else {
                bm = efloat_binade_max(p, emin, emax + 1);
            }
            break;
    }

    return efloat_binade_to_double(bm);
}

/// @brief Returns true if the given EFloat parameters form a valid format
/// (assuming `p = nbits - es >= 2`).
constexpr bool efloat_is_valid(
    prec_t es, prec_t nbits, bool enable_inf, EFloatNanKind nan_kind)
{
    if (es >= nbits || efloat_prec(es, nbits) < 2) {
        return false;
    }
    switch (nan_kind) {
        case EFloatNanKind::IEEE_754:
            // largest-exponent NaNs require an exponent field
            return es != 0;
        case EFloatNanKind::MAX_VAL:
            // the reserved Inf/NaN patterns would consume every value
            return !(es == 0 && enable_inf && efloat_prec(es, nbits) == 2);
        case EFloatNanKind::NEG_ZERO:
        case EFloatNanKind::NONE:
            return true;
    }
    return false;
}

/// @brief Selects the base overflow behavior for an EFloat format. Overflow
/// saturates to `maxval` only when neither infinities nor NaNs are available
/// to represent it; otherwise we let the base produce an infinity (which the
/// `EFloatContext` fixup may then remap to NaN).
constexpr OverflowMode efloat_overflow_mode(bool enable_inf, EFloatNanKind nan_kind) {
    return (!enable_inf && nan_kind == EFloatNanKind::NONE)
        ? OverflowMode::SATURATE
        : OverflowMode::OVERFLOW;
}

} // anonymous namespace

/// @brief "Extended" floating-point rounding context.
///
/// Extends the usual IEEE 754 format with three additional parameters:
/// whether infinities are enabled, how NaNs are encoded, and an exponent
/// offset. See https://uwplse.org/2025/02/17/Small-Floats.html for details.
///
/// This is a reduced port of FPy's `EFloatContext`: it assumes a precision
/// of at least two bits (`p = nbits - es >= 2`) and omits the encode/decode,
/// stochastic rounding, and custom NaN/Inf substitution values. Like the
/// other MPFX contexts, rounding produces a `double`; special values use the
/// native `double` representations of infinity and NaN.
class EFloatContext : public Context {
public:

    /// @brief Constructs an extended floating-point context.
    /// @param es number of exponent bits
    /// @param nbits total number of bits (including sign bit)
    /// @param enable_inf whether infinities are representable
    /// @param nan_kind how NaNs are encoded
    /// @param eoffset exponent offset
    /// @param rm rounding mode
    ///
    /// The parameters are not validated; use `is_valid()` to check that they
    /// form a representable format.
    constexpr EFloatContext(
        prec_t es,
        prec_t nbits,
        bool enable_inf,
        EFloatNanKind nan_kind,
        exp_t eoffset,
        RM rm)
        : Context(
            efloat_prec(es, nbits),
            efloat_emin(es, eoffset) - static_cast<exp_t>(efloat_prec(es, nbits)),
            efloat_max_value(es, nbits, enable_inf, nan_kind, eoffset),
            rm,
            efloat_overflow_mode(enable_inf, nan_kind)),
        es_(es), nbits_(nbits), enable_inf_(enable_inf),
        nan_kind_(nan_kind), eoffset_(eoffset)
    {}

    /// @brief Returns true if this context's parameters form a valid format
    /// (assuming `p = nbits - es >= 2`).
    constexpr bool is_valid() const {
        return efloat_is_valid(es_, nbits_, enable_inf_, nan_kind_);
    }

    /// @brief Gets the number of exponent bits.
    constexpr prec_t es() const { return es_; }

    /// @brief Gets the total number of bits (including sign bit).
    constexpr prec_t nbits() const { return nbits_; }

    /// @brief Whether infinities are representable.
    constexpr bool enable_inf() const { return enable_inf_; }

    /// @brief How NaNs are encoded.
    constexpr EFloatNanKind nan_kind() const { return nan_kind_; }

    /// @brief Gets the exponent offset.
    constexpr exp_t eoffset() const { return eoffset_; }

    /// @brief Gets the minimum normalized exponent of this context.
    constexpr exp_t emin() const { return efloat_emin(es_, eoffset_); }

    /// @brief Gets the maximum normalized exponent of this context.
    constexpr exp_t emax() const { return efloat_emax(es_, eoffset_); }

    /// @brief Rounds `x` according to this context.
    template <flag_mask_t FlagMask = Flags::ALL_FLAGS>
    double round(double x) const {
        return fixup(Context::round<FlagMask>(x));
    }

    /// @brief Rounds `m * 2^exp` according to this context.
    template <flag_mask_t FlagMask = Flags::ALL_FLAGS, signed_integral T>
    double round(T m, exp_t exp) const {
        return fixup(Context::round<FlagMask>(m, exp));
    }

private:

    /// @brief Remaps native infinities/NaNs to the value this format actually
    /// represents (mirrors FPy's `EFloatContext._fixup`).
    double fixup(double y) const {
        if (std::isnan(y) && nan_kind_ == EFloatNanKind::NONE) {
            // NaN is unrepresentable: substitute infinity or `maxval`
            if (enable_inf_) {
                return std::copysign(std::numeric_limits<double>::infinity(), y);
            }
            return std::copysign(*maxval_, y);
        }
        if (std::isinf(y) && !enable_inf_) {
            // infinity is unrepresentable: substitute NaN or `maxval`
            if (nan_kind_ != EFloatNanKind::NONE) {
                return std::copysign(std::numeric_limits<double>::quiet_NaN(), y);
            }
            return std::copysign(*maxval_, y);
        }
        return y;
    }

    /// @brief Number of exponent bits.
    prec_t es_;
    /// @brief Total number of bits (including sign bit).
    prec_t nbits_;
    /// @brief Whether infinities are representable.
    bool enable_inf_;
    /// @brief How NaNs are encoded.
    EFloatNanKind nan_kind_;
    /// @brief Exponent offset.
    exp_t eoffset_;
};

} // namespace mpfx
