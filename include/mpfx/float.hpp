/// @file float.hpp
/// @brief Bit-level operations on floating-point data

#pragma once

#include <bit>
#include <concepts>
#include <tuple>

#include "params.hpp"

namespace mpfx {

template <std::floating_point T>
class bit_float {
public:
    using params_t = float_params<T>::params;
    using uint_t = typename float_params<T>::uint_t;
    static constexpr size_t W = 8 * sizeof(uint_t);

    /// @brief Constructs a default `bit_float` with all bits set to zero.
    constexpr bit_float() : bits_(0) {}

    /// @brief Constructs a `bit_float` from raw bits.
    /// @param bits the raw bits to set
    constexpr bit_float(uint_t bits) : bits_(bits) {}

    /// @brief Constructs a `bit_float` from a floating-point value.
    /// @param value the floating-point value to convert
    constexpr bit_float(T value) : bits_(std::bit_cast<uint_t>(value)) {}

    /// @brief Constructs a `bit_float` encoding `2^exp`.
    /// @param exp the exponent
    static constexpr bit_float make_pow2(exp_t exp) {
        MPFX_DEBUG_ASSERT(exp >= params_t::EXPMIN, "exponent is too small");
        MPFX_DEBUG_ASSERT(exp <= params_t::EMAX, "exponent is too large");

        if (exp < params_t::EMIN) {
            // subnormal result
            const exp_t shift = params_t::EMIN - exp;
            const uint_t mbits = params_t::IMPLICIT1 >> shift;
            return bit_float(mbits);
        } else {
            // normal result
            const uint_t ebits = static_cast<uint_t>(exp + params_t::BIAS);
            return bit_float(ebits << params_t::M);
        }
    }

    /// @brief Returns whether the `bit_float` represents a zero value.
    constexpr bool is_zero() const {
        return (bits_ & ~params_t::SMASK) == 0;
    }

    /// @brief Returns whether the `bit_float` represents an infinity or NaN.
    constexpr bool is_nar() const {
        return (bits_ & params_t::EMASK) == params_t::EMASK;
    }

    /// @brief Returns whether the `bit_float` represents an infinity.
    constexpr bool is_inf() const {
        const bool is_nar = (bits_ & params_t::EMASK) == params_t::EMASK;
        const bool is_zero_mantissa = (bits_ & params_t::MMASK) == 0;
        return is_nar && is_zero_mantissa;
    }

    /// @brief Returns whether the `bit_float` represents a NaN (Not a Number).
    constexpr bool is_nan() const {
        const bool is_nar = (bits_ & params_t::EMASK) == params_t::EMASK;
        const bool is_nonzero_mantissa = (bits_ & params_t::MMASK) != 0;
        return is_nar && is_nonzero_mantissa;
    }

    /// @brief Returns the raw bits of the `bit_float`.
    /// @return the raw bits as an unsigned integer
    constexpr uint_t to_bits() const {
        return bits_;
    }

    /// @brief Converts the `bit_float` back to a floating-point value.
    /// @return the floating-point value represented by the bits
    constexpr T to_float() const {
        return std::bit_cast<T>(bits_);
    }

    /// @brief Returns the sign bit of the `bit_float`.
    /// @return the sign bit as an unsigned integer (0 for positive, 1 for negative)
    constexpr bool s() const {
        return bits_ & params_t::SMASK;
    }

    /// @brief Returns the normalized exponent of the `bit_float`.
    /// @return the normalized exponent as an integer
    constexpr exp_t e() const {
        const uint_t ebits = bits_ & params_t::EMASK;
        if (ebits == 0) {
            // subnormal number
            constexpr size_t min_lz = W - params_t::P;
            const uint_t m = static_cast<uint_t>(bits_ & params_t::MMASK);
            const size_t lz = std::countl_zero(m) - min_lz;
            return params_t::EMIN - static_cast<exp_t>(lz);
        } else {
            // normal number (ignoring Inf and NaN cases)
            MPFX_DEBUG_ASSERT(!is_nar(), "cannot compute exponent for NaN or Inf");
            return static_cast<exp_t>(ebits >> params_t::M) - params_t::BIAS;
        }
    }

    /// @brief Returns the unnormalized exponent of the `bit_float`.
    /// @return the unnormalized exponent as an integer
    constexpr exp_t exp() const {
        const uint_t ebits = bits_ & params_t::EMASK;
        if (ebits == 0) {
            // subnormal number
            return params_t::EXPMIN;
        } else {
            // normal number (ignoring Inf and NaN cases)
            MPFX_DEBUG_ASSERT(!is_nar(), "cannot compute exponent for NaN or Inf");
            const exp_t e = static_cast<exp_t>(ebits >> params_t::M) - params_t::BIAS;
            return e - static_cast<exp_t>(params_t::P - 1);
        }
    }

    /// @brief Returns the integer significand of the `bit_float`,
    /// including the implicit leading bit for normal numbers.
    /// @return the significand as an unsigned integer
    constexpr uint_t c() const {
        const uint_t ebits = bits_ & params_t::EMASK;
        const uint_t m = bits_ & params_t::MMASK;
        if (ebits == 0) {
            // subnormal number
            return m;
        } else {
            // normal number (ignoring Inf and NaN cases)
            MPFX_DEBUG_ASSERT(!is_nar(), "cannot compute significand for NaN or Inf");
            return m | params_t::IMPLICIT1; // add implicit leading bit
        }
    }

    /// @brief Returns the (true) precision of the `bit_float`.
    /// @return the precision as an integer
    constexpr prec_t p() const {
        const uint_t ebits = bits_ & params_t::EMASK;
        if (ebits == 0) {
            // subnormal number
            constexpr size_t min_lz = W - params_t::P;
            const uint_t m = static_cast<uint_t>(bits_ & params_t::MMASK);
            const size_t lz = std::countl_zero(m) - min_lz;
            return params_t::P - static_cast<prec_t>(lz);
        } else {
            // normal number (ignoring Inf and NaN cases)
            MPFX_DEBUG_ASSERT(!is_nar(), "cannot compute precision for NaN or Inf");
            return params_t::P;
        }
    }

    /// @brief Return a triple (s, exp, c) representing
    /// the sign, exponent, and significand of the `bit_float`.
    /// @return a tuple containing the sign bit, exponent, and significand
    constexpr std::tuple<bool, exp_t, uint_t> unpack() const {
        return { s(), exp(), c() };
    }

    /// @brief Extracts the bit at position `n`.
    /// @param n the bit position to extract.
    /// @return the bit at position `n` as a boolean
    constexpr bool bit(exp_t n) const {
        MPFX_DEBUG_ASSERT(!is_nar(), "cannot compute significand for NaN or Inf");

        // fast path: out of range
        if (n < params_t::EXPMIN || n > params_t::EMAX) {
            return false;
        }

        // compute the range of the mantissa bits
        const exp_t exp = this->exp();
        if (n < exp || n >= exp + static_cast<exp_t>(params_t::P)) {
            return false;
        }

        // extract the bit from the significand
        const uint_t c = this->c();
        return (c >> (n - exp)) & 0x1;
    }

    /// @brief Splits this `bit_float` at digit position `n`.
    /// @param n the digit position to split at
    /// @return a pair of `bit_float`s representing the high and low parts
    constexpr std::pair<bit_float, bit_float> split(exp_t n) const {
        MPFX_DEBUG_ASSERT(!is_nar(), "cannot compute exponent for NaN or Inf");

        // fast path: zero
        if (is_zero()) {
            return { *this, bit_float() };
        }

        // compute the normalized exponent
        const uint_t ebits = bits_ & params_t::EMASK;
        const exp_t e = (ebits == 0)
            ? params_t::EMIN
            : (static_cast<exp_t>(ebits >> params_t::M) - params_t::BIAS);

        // if split point is at or above `e`, then all digits
        // are in the low part, and the high part is zero
        if (n >= e) {
            // preseve the sign
            const uint_t sbits = bits_ & params_t::SMASK;
            return { bit_float(sbits), *this };
        }

        // if the split point is at or below `e - P`, then all digits
        // are in the high part, and the low part is zero
        const exp_t exp = e - static_cast<exp_t>(params_t::P);
        if (exp > n) {
            return { *this, bit_float() };
        };

        // otherwise, we need to split the bits
        const prec_t p_high = static_cast<prec_t>(e - n);
        const prec_t p_low = params_t::P - p_high;
        const prec_t p_mask = bitmask<uint_t>(p_low);

        // split the mantissa bits into high and low parts
        const uint_t se = bits_ & ~params_t::MMASK;
        const uint_t m = bits_ & params_t::MMASK;
        const uint_t c = (ebits == 0) ? m : (m | params_t::IMPLICIT1); // add implicit leading bit for normal numbers
        const uint_t c_high = c & ~p_mask;
        uint_t c_low = c & p_mask;

        // reform the high part
        const uint_t high = se | (c_high & params_t::MMASK);

        // fast path: low == 0
        if (c_low == 0) {
            return { bit_float(high), bit_float() };
        }

        // normalize the low part
        constexpr size_t min_lz = W - params_t::P;
        const size_t lz = std::countl_zero(c_low) - min_lz;
        const exp_t e_low = e - static_cast<exp_t>(lz);
        c_low <<= lz;

        // case split on exponent
        uint_t low;
        if (e_low < params_t::EMIN) {
            // subnormal number
            const size_t offset = static_cast<size_t>(params_t::EMIN - e_low);
            c_low >>= offset;

            const uint_t sbits_low = bits_ & params_t::SMASK;
            const uint_t mbits_low = static_cast<uint_t>(c_low & params_t::MMASK);
            low = sbits_low | mbits_low;
        } else {
            // normal
            const uint_t sbits_low = bits_ & params_t::SMASK;
            const uint_t ebits_low = static_cast<uint_t>(e_low + params_t::BIAS);
            const uint_t mbits_low = static_cast<uint_t>(c_low & params_t::MMASK);
            low = sbits_low | (ebits_low << params_t::M) | mbits_low;
        }

        return { bit_float(high), bit_float(low) };
    }

    private:
        uint_t bits_;
};

} // end namespace mpfx
