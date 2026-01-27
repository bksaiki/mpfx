#pragma once

#include <floppy_float.h>
#include <vfpu.h>

#include "types.hpp"

namespace mpfx {

namespace engine_ff {

/// @brief Computes `x + y` using FloppyFloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "add: requested precision exceeds double-precision capability"
    );

    // initialize FloppyFloat instance
    static FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTowardZero;

    // Perform round-to-zero addition with round-to-odd fix-up
    double z = ff.Add(x, y);
    if (ff.inexact) {
        uint64_t b = std::bit_cast<uint64_t>(z);
        b |= 1; // set LSB
        z = std::bit_cast<double>(b);
        ff.inexact = false;
    }

    return z;
}

/// @brief Computes `x - y` using FloppyFloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "sub: requested precision exceeds double-precision capability"
    );

    // initialize FloppyFloat instance
    static FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTowardZero;

    // Perform round-to-zero subtraction with round-to-odd fix-up
    double z = ff.Sub(x, y);
    if (ff.inexact) {
        uint64_t b = std::bit_cast<uint64_t>(z);
        b |= 1; // set LSB
        z = std::bit_cast<double>(b);
        ff.inexact = false;
    }

    return z;
}

/// @brief Computes `x * y` using FloppyFloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "mul: requested precision exceeds double-precision capability"
    );

    // initialize FloppyFloat instance
    static FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTowardZero;

    // Perform round-to-zero multiplication with round-to-odd fix-up
    double z = ff.Mul(x, y);
    if (ff.inexact) {
        uint64_t b = std::bit_cast<uint64_t>(z);
        b |= 1; // set LSB
        z = std::bit_cast<double>(b);
        ff.inexact = false;
    }

    return z;
}

/// @brief Computes `x / y` using FloppyFloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double div(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "div: requested precision exceeds double-precision capability"
    );

    // initialize FloppyFloat instance
    static FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTowardZero;

    // Perform round-to-zero division with round-to-odd fix-up
    double z = ff.Div(x, y);
    if (ff.inexact) {
        uint64_t b = std::bit_cast<uint64_t>(z);
        b |= 1; // set LSB
        z = std::bit_cast<double>(b);
        ff.inexact = false;
    }

    return z;
}

/// @brief Computes `sqrt(x)` using FloppyFloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double sqrt(double x, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "sqrt: requested precision exceeds double-precision capability"
    );

    // initialize FloppyFloat instance
    static FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTowardZero;

    // Perform round-to-zero square root with round-to-odd fix-up
    double z = ff.Sqrt(x);
    if (ff.inexact) {
        uint64_t b = std::bit_cast<uint64_t>(z);
        b |= 1; // set LSB
        z = std::bit_cast<double>(b);
        ff.inexact = false;
    }

    return z;
}

/// @brief Computes `x * y + z` using FloppyFloat.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
inline double fma(double x, double y, double z, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    MPFX_DEBUG_ASSERT(
        p <= 53,
        "fma: requested precision exceeds double-precision capability"
    );

    // initialize FloppyFloat instance
    FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTowardZero;

    // Perform round-to-zero FMA with round-to-odd fix-up
    double result = ff.Fma(x, y, z);
    if (ff.inexact) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
        ff.inexact = false;
    }

    return result;
}


} // end namespace engine_ff

} // end namespace mpfx
