#pragma once

#include "context_mpb.hpp"
#include "params.hpp"
#include "types.hpp"

namespace mpfx {

inline prec_t __ieee754_prec(prec_t es, prec_t nbits) {
    return nbits - es;
}

inline exp_t __ieee754_emax(prec_t es) {
    return static_cast<exp_t>(bitmask<uint32_t>(es - 1));
}

inline exp_t __ieee754_emin(prec_t es) {
    return 1 - __ieee754_emax(es);
}

inline double __ieee754_max_value(prec_t es, prec_t nbits) {
    using FP64 = ieee754_consts<11, 64>; // IEEE 754 double precision

    // format parameters
    const prec_t prec = __ieee754_prec(es, nbits);
    const exp_t emax = __ieee754_emax(es);

    // encode maximum value as double
    const uint64_t mbits = bitmask<mant_t>(prec - 1) << (FP64::P - prec);
    const uint64_t ebits = static_cast<uint64_t>(emax + FP64::BIAS) << FP64::M;
    const uint64_t bits = ebits | mbits;
    return std::bit_cast<double>(bits);
}

/// @brief IEEE 754 floating-point rounding context.
///
/// This context implements the usual IEEE 754 semantics for floating-point
/// arithmetic, including exponent bounds and overflow handling.
class IEEE754Context : public MPBContext {
private:
    /// @brief Number of exponent bits.
    prec_t es_;
    /// @brief Total number of bits (including sign bit).
    prec_t nbits_;

public:

    /// @brief Constructs an IEEE 754 context.
    /// @param es number of exponent bits
    /// @param nbits total number of bits (including sign bit)
    /// @param rm rounding mode
    inline IEEE754Context(prec_t es, prec_t nbits, RM rm)
        : MPBContext(
            __ieee754_prec(es, nbits),
            __ieee754_emin(es), rm,
            __ieee754_max_value(es, nbits)),
        es_(es), nbits_(nbits) {}

    /// @brief Gets the number of exponent bits.
    inline prec_t es() const {
        return es_;
    }

    /// @brief Gets the total number of bits (including sign bit).
    inline prec_t nbits() const {
        return nbits_;
    }
};

} // namespace mpfx
