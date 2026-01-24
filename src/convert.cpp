#include <cmath>
#include <optional>
#include <tuple>

#include "mpfx/convert.hpp"
#include "mpfx/params.hpp"

namespace mpfx {

std::tuple<int64_t, exp_t> to_fixed(double x) {
    using FP = ieee754_consts<11, 64>; // double precision
    FPY_ASSERT(std::isfinite(x), "to_fixed: input must be finite");

    // fast path: zero
    if (x == 0.0) {
        return std::make_tuple(0, FP::EXPMIN);
    }

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const bool s = (b >> (FP::N - 1)) != 0;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decode (unsigned) floating-point data
    exp_t exp; // unnormalized exponent
    mant_t c;  // integer significand
    if (ebits == 0) {
        // subnormal
        exp = FP::EXPMIN;
        c = mbits;
    } else {
        // normal (assuming no infinity or NaN)
        const exp_t e = static_cast<exp_t>(ebits) - FP::BIAS;
        exp = e - static_cast<exp_t>(FP::M);
        c = FP::IMPLICIT1 | mbits;
    }

    // minimize the precision of the significand
    const auto tz = std::countr_zero(c);
    c >>= tz;
    exp += static_cast<exp_t>(tz);

    // apply sign
    const int64_t m = s ? -static_cast<int64_t>(c) : static_cast<int64_t>(c);
    return std::make_tuple(m, exp);
}


static std::pair<mant_t, exp_t> normalize(
    const mant_t c,
    const exp_t exp,
    const std::optional<prec_t>& p,
    const std::optional<exp_t>& n
) {
    // special case: zero
    if (c == 0) {
        if (n.has_value()) {
            return { 0, *n + 1 };
        } else {
            return { 0, exp };
        }
    }

    // case split on precision
    exp_t new_exp, shift;
    if (p.has_value()) {
        // requesting maximum precision
        shift = static_cast<exp_t>(*p) - static_cast<exp_t>(std::bit_width(c));
        new_exp = exp - shift;
        if (n.has_value() && new_exp <= *n) {
            // requesting lower bound on digits
            // exponent is too small, adjust if necessary
            const exp_t expmin = *n + 1;
            const exp_t adjust = expmin - new_exp;
            shift -= adjust;
            new_exp += adjust;
        }
    } else {
        // no maximum precision
        if (n.has_value() && exp <= *n) {
            // requesting lower bound on digits
            const exp_t expmin = *n + 1;
            shift = expmin - exp;
            new_exp = exp + shift;
        } else {
            // no adjustment needed
            shift = 0;
            new_exp = exp;
        }
    }

    // compute new significand `c`
    if (shift == 0) {
        // no shifting required
        return { c, new_exp };
    } else if (shift > 0) {
        // shift left by a non-negative amount
        const prec_t p_new = std::bit_width(c) + static_cast<prec_t>(shift);
        FPY_DEBUG_ASSERT(p_new <= 64, "normalize: precision exceeds 64 bits");
        return { c << shift, new_exp };
    } else {
        // shift right by a positive amount
        const mant_t c_lost = c & bitmask<mant_t>(-shift);
        FPY_DEBUG_ASSERT(c_lost == 0, "normalize: losing digits");
        return { c >> -shift, new_exp };
    }
}


double digits_to_double(bool s, exp_t exp, mant_t c) {
    using FP = ieee754_consts<11, 64>;

    // special case: zero
    if (c == 0) {
        return s ? -0.0 : 0.0;
    }

    // case split on normalization
    const auto [c_norm, exp_norm] = normalize(c, exp, FP::P, FP::EXPMIN - 1);
    const prec_t p = std::bit_width(c_norm);
    const uint64_t ebits = p == FP::P ? (exp_norm - FP::EXPMIN + 1) : 0;
    const uint64_t mbits = c_norm & bitmask<mant_t>(FP::M); 

    // cast to double
    const uint64_t b = (ebits << FP::M) | mbits;
    const double r = std::bit_cast<double>(b);
    return s ? -r : r;
}

} // end namespace mpfx
