#include "mpfx/context_mpb.hpp"

namespace mpfx {

MPBContext::MPBContext(prec_t prec, exp_t emin, double maxval, RM rm)
    : Context(prec, emin - static_cast<exp_t>(prec), maxval, rm)
    , emin_(emin) {
    // Compute emax_ from maxval
    using FP = float_params<double>::params;
    emax_ = maxval == 0.0 ? FP::EMIN : static_cast<exp_t>(std::ilogb(maxval));
}

} // namespace mpfx
