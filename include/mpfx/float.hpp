/// @file float.hpp
/// @brief Bit-level operations on floating-point data

#pragma once

#include <bit>
#include <concepts>

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

    /// @brief Returns whether the `bit_float` represents a zero value.
    constexpr bool is_zero() const {
        return (bits_ & ~params_t::SMASK) == 0;
    }

    /// @brief Returns whether the `bit_float` represents an infinity or NaN.
    constexpr bool is_nar() const {
        return (bits_ & params_t::EMASK) == params_t::EMASK;
    }

    /// @brief Returns the raw bits of the `bit_float`.
    /// @return the raw bits as an unsigned integer
    constexpr uint_t bits() const { return bits_; }

    /// @brief Converts the `bit_float` back to a floating-point value.
    /// @return the floating-point value represented by the bits
    constexpr T to_float() const { return std::bit_cast<T>(bits_); }

    /// @brief Returns the (true) precision of the `bit_float`.
    /// @return the precision as an integer
    constexpr prec_t p() const {
        const uint_t ebits = bits_ & params_t::EMASK;
        if (ebits == 0) {
            // subnormal number
            const size_t min_lz = W - params_t::P;
            const uint_t m = static_cast<uint_t>(bits_ & params_t::MMASK);
            const size_t lz = std::countl_zero(m) - min_lz;
            return params_t::P - static_cast<prec_t>(lz);
        } else {
            // normal number (ignoring Inf and NaN cases)
            return params_t::P;
        }
    }

    /// @brief Returns the normalized exponent of the `bit_float`.
    /// @return the normalized exponent as an integer
    constexpr exp_t e() const {
        const uint_t ebits = bits_ & params_t::EMASK;
        if (ebits == 0) {
            // subnormal number
            return params_t::EMIN;
        } else {
            // normal number (ignoring Inf and NaN cases)
            return static_cast<exp_t>(ebits >> params_t::M) - params_t::BIAS;
        }
    }

    /// @brief Returns the unnormalized exponent of the `bit_float`.
    /// @return the unnormalized exponent as an integer
    constexpr exp_t exp() const {
        return e() - static_cast<exp_t>(params_t::P - 1);
    }

    private:
        uint_t bits_;
};

} // end namespace mpfx
