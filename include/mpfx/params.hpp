#pragma once

#include <concepts>

#include "types.hpp"

namespace mpfx {

/// @brief Type trait defining IEEE 754 format constants
/// @tparam E bitwidth of the exponent field
/// @tparam N total bitwidth of the format
template <uint64_t E_, uint64_t N_>
struct ieee754_params {
    static constexpr uint64_t E = E_;
    static constexpr uint64_t N = N_;

    MPFX_STATIC_ASSERT(E >= 2, "Invalid IEEE 754 format");
    MPFX_STATIC_ASSERT(N >= E + 2, "Invalid IEEE 754 format");
    MPFX_STATIC_ASSERT(N <= 64, "Exceeded maximum supported IEEE 754 format");

    static constexpr prec_t P = static_cast<prec_t>(N - E);
    static constexpr prec_t M = P - 1;

    static constexpr exp_t EMAX = static_cast<exp_t>(bitmask<uint32_t>(E - 1));
    static constexpr exp_t EMIN = 1 - EMAX;
    static constexpr exp_t EXPMAX = EMAX - static_cast<exp_t>(P) + 1;
    static constexpr exp_t EXPMIN = EMIN - static_cast<exp_t>(P) + 1;
    static constexpr exp_t BIAS = EMAX;

    static constexpr uint64_t SMASK = 1ULL << (N - 1);
    static constexpr uint64_t EMASK = bitmask<uint64_t>(E) << M;
    static constexpr uint64_t MMASK = bitmask<uint64_t>(M);
    static constexpr uint64_t EONES = bitmask<uint64_t>(E);
    static constexpr mant_t IMPLICIT1 = 1ULL << M;
};

/// @brief Type trait providing IEEE 754 format parameters and constants.
/// @tparam T floating-point type
template <std::floating_point T>
struct float_params {};

template <>
struct float_params<float> {
    using params = ieee754_params<8, 32>;
    using limits = std::numeric_limits<float>;
    using uint_t = uint32_t;
    using int_t = int32_t;
};

template <>
struct float_params<double> {
    using params = ieee754_params<11, 64>;
    using limits = std::numeric_limits<double>;
    using uint_t = uint64_t;
    using int_t = int64_t;
};

} // namespace mpfx

