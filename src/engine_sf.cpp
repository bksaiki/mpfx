#include <bit>
#include <concepts>
#include <type_traits>

#include "mpfx/engine_sf.hpp"
#include "mpfx/params.hpp"

extern "C" {
#include <softfloat.h>
}

namespace mpfx {

namespace engine_sf {

// Conversions between native floating-point and SoftFloat's bit-wrapped types.
static float64_t to_sf(double x) { return { std::bit_cast<uint64_t>(x) }; }
static float32_t to_sf(float x) { return { std::bit_cast<uint32_t>(x) }; }
static double from_sf(float64_t x) { return std::bit_cast<double>(x.v); }
static float from_sf(float32_t x) { return std::bit_cast<float>(x.v); }

template <std::floating_point T>
T add(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "add: requested precision exceeds the container type's capability"
    );

    softfloat_roundingMode = softfloat_round_odd;
    if constexpr (std::is_same_v<T, float>) {
        return from_sf(f32_add(to_sf(x), to_sf(y)));
    } else {
        return from_sf(f64_add(to_sf(x), to_sf(y)));
    }
}

template <std::floating_point T>
T sub(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "sub: requested precision exceeds the container type's capability"
    );

    softfloat_roundingMode = softfloat_round_odd;
    if constexpr (std::is_same_v<T, float>) {
        return from_sf(f32_sub(to_sf(x), to_sf(y)));
    } else {
        return from_sf(f64_sub(to_sf(x), to_sf(y)));
    }
}

template <std::floating_point T>
T mul(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "mul: requested precision exceeds the container type's capability"
    );

    softfloat_roundingMode = softfloat_round_odd;
    if constexpr (std::is_same_v<T, float>) {
        return from_sf(f32_mul(to_sf(x), to_sf(y)));
    } else {
        return from_sf(f64_mul(to_sf(x), to_sf(y)));
    }
}

template <std::floating_point T>
T div(T x, T y, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "div: requested precision exceeds the container type's capability"
    );

    softfloat_roundingMode = softfloat_round_odd;
    if constexpr (std::is_same_v<T, float>) {
        return from_sf(f32_div(to_sf(x), to_sf(y)));
    } else {
        return from_sf(f64_div(to_sf(x), to_sf(y)));
    }
}

template <std::floating_point T>
T sqrt(T x, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "sqrt: requested precision exceeds the container type's capability"
    );

    softfloat_roundingMode = softfloat_round_odd;
    if constexpr (std::is_same_v<T, float>) {
        return from_sf(f32_sqrt(to_sf(x)));
    } else {
        return from_sf(f64_sqrt(to_sf(x)));
    }
}

template <std::floating_point T>
T fma(T x, T y, T z, prec_t p) {
    MPFX_DEBUG_ASSERT(
        p <= float_params<T>::params::P,
        "fma: requested precision exceeds the container type's capability"
    );

    softfloat_roundingMode = softfloat_round_odd;
    if constexpr (std::is_same_v<T, float>) {
        return from_sf(f32_mulAdd(to_sf(x), to_sf(y), to_sf(z)));
    } else {
        return from_sf(f64_mulAdd(to_sf(x), to_sf(y), to_sf(z)));
    }
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

} // end namespace engine_sf

} // end namespace mpfx
