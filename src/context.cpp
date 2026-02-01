#include "mpfx/context.hpp"

namespace mpfx {

Context::Context(prec_t p, const std::optional<exp_t>& n, const std::optional<double>& maxval, RM rm)
    : p_(p), n_(n), maxval_(maxval), rm_(rm) {
    using FP = float_params<double>::params;

    // Validate and compute maxval_odd_ if maxval is present
    if (maxval_.has_value()) {
        const double maxval = *maxval_;
        MPFX_ASSERT(!std::signbit(maxval), "maxval must be non-negative");
        MPFX_ASSERT(maxval == mpfx::round<Flags::NO_FLAGS>(maxval, p_, n_, rm_), "maxval must be finite");

        // Check if the maximum value is odd
        const uint64_t bits = std::bit_cast<uint64_t>(maxval);
        const int pth_bit_pos = static_cast<int>(FP::M) - static_cast<int>(p_) + 1;
        maxval_is_odd_ = (pth_bit_pos >= 0) && ((bits >> pth_bit_pos) & 1);
    }
}

} // namespace mpfx
