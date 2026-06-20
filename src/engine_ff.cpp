#include "mpfx/engine_ff.hpp"

#include <bit>
#include <concepts>

#include "mpfx/params.hpp"

#include <floppy_float.h>
#include <vfpu.h>

namespace mpfx {

namespace engine_ff {

// Shared FloppyFloat instance for all operations
static FloppyFloat ff = []() {
    FloppyFloat f;
    f.rounding_mode = Vfpu::kRoundTowardZero;
    return f;
}();

// FloppyFloat rounds toward zero; jam the LSB on inexact results to obtain
// round-to-odd, then clear the sticky inexact flag for the next operation.
template <std::floating_point T>
static T rto_fixup(T z) {
    using U = typename float_params<T>::uint_t;
    if (ff.inexact) {
        U b = std::bit_cast<U>(z);
        b |= 1; // set LSB
        z = std::bit_cast<T>(b);
        ff.inexact = false;
    }
    return z;
}

template <std::floating_point T>
T add(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "add: requested precision exceeds the container type's capability"
    );
    return rto_fixup(ff.Add(x, y));
}

template <std::floating_point T>
T sub(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "sub: requested precision exceeds the container type's capability"
    );
    return rto_fixup(ff.Sub(x, y));
}

template <std::floating_point T>
T mul(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "mul: requested precision exceeds the container type's capability"
    );
    return rto_fixup(ff.Mul(x, y));
}

template <std::floating_point T>
T div(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "div: requested precision exceeds the container type's capability"
    );
    return rto_fixup(ff.Div(x, y));
}

template <std::floating_point T>
T sqrt(T x, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "sqrt: requested precision exceeds the container type's capability"
    );
    return rto_fixup(ff.Sqrt(x));
}

template <std::floating_point T>
T fma(T x, T y, T z, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "fma: requested precision exceeds the container type's capability"
    );
    return rto_fixup(ff.Fma(x, y, z));
}

// Explicit instantiations for the supported container types.
template float add<float>(float, float, prec_t);
template double add<double>(double, double, prec_t);
template float sub<float>(float, float, prec_t);
template double sub<double>(double, double, prec_t);
template float mul<float>(float, float, prec_t);
template double mul<double>(double, double, prec_t);
template float div<float>(float, float, prec_t);
template double div<double>(double, double, prec_t);
template float sqrt<float>(float, prec_t);
template double sqrt<double>(double, prec_t);
template float fma<float>(float, float, float, prec_t);
template double fma<double>(double, double, double, prec_t);

} // end namespace engine_ff

} // end namespace mpfx
