/**
 * @file test_exhaustive.cpp
 * @brief Exhaustive differential test of every MPFX engine against MPFR for
 *        small IEEE 754-style formats.
 *
 * For each format (es, nbits) with nbits <= 8, every finite value and both
 * infinities are enumerated, and every operation is evaluated on every input
 * tuple under all seven target rounding modes (RNE, RNA, RTP, RTN, RTZ, RAZ,
 * RTO). Results are compared bit-for-bit (NaNs compared as a class) against an
 * MPFR oracle that emulates the format's subnormal range and overflow
 * threshold. Both container types (`double` and `float`) and all four
 * multi-precision engines (FP_RTO, EFT, SoftFloat, FloppyFloat) are covered.
 *
 * MPFR has no ties-away or round-to-odd mode, so those two oracles are derived:
 *   RNA: equals RNE unless the exact result is a midpoint of the target
 *        quantisation, in which case it equals RAZ.
 *   RTO: RTZ at the target precision, then the LSB is forced odd if inexact;
 *        the result overflows only if it exceeds the format's maximum.
 *
 * Exhaustive FMA over an 8-bit format is about 2^24 triples per rounding mode,
 * so FMA stops at nbits <= 7; the randomized tests in test_ops.cpp cover the
 * rest.
 */

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

#include <mpfr.h>
#include <mpfx.hpp>
#include <gtest/gtest.h>

namespace {

///////////////////////////////////////////////////////////
// Formats and value enumeration

struct Format {
    mpfx::prec_t es;
    mpfx::prec_t nbits;

    mpfx::prec_t p() const { return nbits - es; }
    mpfx::exp_t emax() const { return (mpfx::exp_t(1) << (es - 1)) - 1; }
    mpfx::exp_t emin() const { return 1 - emax(); }
    // exponent of the smallest subnormal, i.e. the format's fixed quantum
    mpfx::exp_t qexp() const { return emin() - p() + 1; }
    double maxval() const { return std::ldexp(2.0 - std::ldexp(1.0, 1 - p()), emax()); }

    // MPFR's exponent convention is 0.1xxx * 2^E, so shift by one relative
    // to IEEE's 1.xxx * 2^e.
    mpfr_exp_t mpfr_emin() const { return qexp() + 1; }
    mpfr_exp_t mpfr_emax() const { return emax() + 1; }

    std::string name() const { return "E" + std::to_string(es) + "M" + std::to_string(p() - 1); }
};

// Every finite value of the format plus both infinities (NaNs excluded).
std::vector<double> enumerate(const Format& f) {
    std::vector<double> vals;
    const uint64_t half = uint64_t(1) << (f.p() - 1);
    for (bool s : {false, true}) {
        vals.push_back(mpfx::make_float<double>(s, 0, 0));
        for (uint64_t c = 1; c < half; c++) {
            vals.push_back(mpfx::make_float<double>(s, f.qexp(), c));
        }
        for (mpfx::exp_t e = f.emin(); e <= f.emax(); e++) {
            for (uint64_t c = half; c < 2 * half; c++) {
                vals.push_back(mpfx::make_float<double>(s, e - f.p() + 1, c));
            }
        }
        const double inf = std::numeric_limits<double>::infinity();
        vals.push_back(s ? -inf : inf);
    }
    return vals;
}

///////////////////////////////////////////////////////////
// MPFR oracle

enum class Op { ADD, SUB, MUL, DIV, SQRT, FMA };

const std::vector<Op> BINARY_OPS = {Op::ADD, Op::SUB, Op::MUL, Op::DIV};

const std::vector<mpfx::RM> ALL_MODES = {
    mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP, mpfx::RM::RTN,
    mpfx::RM::RTZ, mpfx::RM::RAZ, mpfx::RM::RTO,
};

const char* op_name(Op op) {
    switch (op) {
        case Op::ADD: return "add";
        case Op::SUB: return "sub";
        case Op::MUL: return "mul";
        case Op::DIV: return "div";
        case Op::SQRT: return "sqrt";
        case Op::FMA: return "fma";
    }
    return "?";
}

const char* rm_name(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE: return "RNE";
        case mpfx::RM::RNA: return "RNA";
        case mpfx::RM::RTP: return "RTP";
        case mpfx::RM::RTN: return "RTN";
        case mpfx::RM::RTZ: return "RTZ";
        case mpfx::RM::RAZ: return "RAZ";
        case mpfx::RM::RTO: return "RTO";
        case mpfx::RM::RTE: return "RTE";
    }
    return "?";
}

// Scoped MPFR exponent range.
class ExpRange {
public:
    ExpRange(mpfr_exp_t emin, mpfr_exp_t emax)
        : saved_min_(mpfr_get_emin()), saved_max_(mpfr_get_emax()) {
        mpfr_set_emin(emin);
        mpfr_set_emax(emax);
    }
    ~ExpRange() {
        mpfr_set_emin(saved_min_);
        mpfr_set_emax(saved_max_);
    }
private:
    mpfr_exp_t saved_min_, saved_max_;
};

class Oracle {
public:
    Oracle() : default_emax_(mpfr_get_emax()) {
        mpfr_init2(x_, 53);
        mpfr_init2(y_, 53);
        mpfr_init2(z_, 53);
        mpfr_init2(hi_, 512);
        mpfr_init2(r_, 2);
    }
    ~Oracle() {
        mpfr_clears(x_, y_, z_, hi_, r_, (mpfr_ptr) 0);
    }

    double eval(Op op, double x, double y, double z, const Format& f, mpfx::RM rm) {
        mpfr_set_d(x_, x, MPFR_RNDN);
        mpfr_set_d(y_, y, MPFR_RNDN);
        mpfr_set_d(z_, z, MPFR_RNDN);
        switch (rm) {
            case mpfx::RM::RNE: return direct(op, f, MPFR_RNDN, true);
            case mpfx::RM::RTP: return direct(op, f, MPFR_RNDU, true);
            case mpfx::RM::RTN: return direct(op, f, MPFR_RNDD, true);
            case mpfx::RM::RTZ: return direct(op, f, MPFR_RNDZ, true);
            case mpfx::RM::RAZ: return direct(op, f, MPFR_RNDA, true);
            case mpfx::RM::RNA: return ties_away(op, f);
            case mpfx::RM::RTO: return to_odd(op, f);
            default: ADD_FAILURE() << "unsupported mode"; return 0.0;
        }
    }

private:
    // Rounds op(x, y, z) with MPFR at the format's precision, with the subnormal
    // range emulated. `bound_emax` selects IEEE overflow handling; otherwise the
    // exponent range is unbounded above. Add/sub/mul/fma are exact at 512 bits
    // and are rounded from that exact value, which also sidesteps mpfr_fma's
    // rejection of very narrow exponent ranges. Div/sqrt call MPFR directly.
    double direct(Op op, const Format& f, mpfr_rnd_t rnd, bool bound_emax) {
        const bool exact_path = (op != Op::DIV && op != Op::SQRT);
        if (exact_path) {
            const int t_hi = apply(hi_, op, MPFR_RNDZ);  // default (huge) range
            EXPECT_EQ(t_hi, 0) << "exact intermediate lost precision";
        }
        ExpRange range(f.mpfr_emin(), bound_emax ? f.mpfr_emax() : default_emax_);
        mpfr_set_prec(r_, f.p());
        // mpfr_set does not clamp to the current exponent range; check_range
        // applies the overflow/underflow rules the arithmetic functions apply.
        int t = exact_path ? mpfr_set(r_, hi_, rnd) : apply(r_, op, rnd);
        t = mpfr_check_range(r_, t, rnd);
        t = mpfr_subnormalize(r_, t, rnd);
        last_inexact_ = (t != 0);
        return mpfr_get_d(r_, MPFR_RNDN);
    }

    int apply(mpfr_ptr r, Op op, mpfr_rnd_t rnd) {
        switch (op) {
            case Op::ADD: return mpfr_add(r, x_, y_, rnd);
            case Op::SUB: return mpfr_sub(r, x_, y_, rnd);
            case Op::MUL: return mpfr_mul(r, x_, y_, rnd);
            case Op::DIV: return mpfr_div(r, x_, y_, rnd);
            case Op::SQRT: return mpfr_sqrt(r, x_, rnd);
            case Op::FMA: return mpfr_fma(r, x_, y_, z_, rnd);
        }
        return 0;
    }

    // True if `v` is representable with `p` bits and quantum 2^qexp.
    bool representable(mpfr_srcptr v, mpfx::prec_t p, mpfx::exp_t qexp) {
        if (!mpfr_number_p(v)) return true;
        ExpRange range(qexp + 1, default_emax_);
        mpfr_set_prec(r_, p);
        int t = mpfr_set(r_, v, MPFR_RNDZ);
        t = mpfr_check_range(r_, t, MPFR_RNDZ);
        t = mpfr_subnormalize(r_, t, MPFR_RNDZ);
        return t == 0 && mpfr_equal_p(r_, v);
    }

    // RNA agrees with RNE except at exact midpoints, where it agrees with RAZ.
    double ties_away(Op op, const Format& f) {
        const double r_ne = direct(op, f, MPFR_RNDN, true);
        const int t_hi = apply(hi_, op, MPFR_RNDZ);  // exact for add/sub/mul/fma
        if (t_hi != 0) {
            return r_ne;  // not a dyadic rational, hence not a midpoint
        }
        const bool midpoint = !representable(hi_, f.p(), f.qexp())
                           && representable(hi_, f.p() + 1, f.qexp() - 1);
        return midpoint ? direct(op, f, MPFR_RNDA, true) : r_ne;
    }

    // RTO: truncate at the target precision, force the LSB odd if inexact, then
    // apply the format's overflow threshold. The quantum below the normal range
    // is fixed, so the LSB is taken relative to the format's actual spacing.
    double to_odd(Op op, const Format& f) {
        double r = direct(op, f, MPFR_RNDZ, false);
        if (last_inexact_ && std::isfinite(r)) {
            double mag = std::fabs(r);
            double q;
            if (mag < std::ldexp(1.0, f.emin())) {
                q = std::ldexp(1.0, f.qexp());
            } else {
                q = std::ldexp(1.0, std::ilogb(mag) - f.p() + 1);
            }
            const double k = mag / q;  // exact integer
            if (std::fmod(k, 2.0) == 0.0) {
                mag += q;
            }
            r = std::copysign(mag, r);
        }
        if (std::isfinite(r) && std::fabs(r) > f.maxval()) {
            r = std::copysign(std::numeric_limits<double>::infinity(), r);
        }
        return r;
    }

    mpfr_t x_, y_, z_, hi_, r_;
    mpfr_exp_t default_emax_;
    bool last_inexact_ = false;
};

///////////////////////////////////////////////////////////
// MPFX under test

template <mpfx::Engine E, std::floating_point T>
T run_op(Op op, T x, T y, T z, const mpfx::Context& ctx) {
    switch (op) {
        case Op::ADD: return mpfx::add<E>(x, y, ctx);
        case Op::SUB: return mpfx::sub<E>(x, y, ctx);
        case Op::MUL: return mpfx::mul<E>(x, y, ctx);
        case Op::DIV: return mpfx::div<E>(x, y, ctx);
        case Op::SQRT: return mpfx::sqrt<E>(x, ctx);
        case Op::FMA: return mpfx::fma<E>(x, y, z, ctx);
    }
    return T(0);
}

const char* engine_name(mpfx::Engine e) {
    switch (e) {
        case mpfx::Engine::FP_RTO: return "FP_RTO";
        case mpfx::Engine::EFT: return "EFT";
        case mpfx::Engine::SOFTFLOAT: return "SoftFloat";
        case mpfx::Engine::FFLOAT: return "FloppyFloat";
        default: return "?";
    }
}

template <std::floating_point T>
bool same_value(T a, T b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::bit_cast<std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>>(a)
        == std::bit_cast<std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>>(b);
}

// Checks one engine/container against the oracle on every input tuple of `op`.
// Reports at most a few mismatches per (format, op, mode) to keep logs short.
template <mpfx::Engine E, std::floating_point T>
void check(Op op, const Format& f, mpfx::RM rm, const std::vector<double>& vals,
           Oracle& oracle, size_t& mismatches, size_t& cases) {
    const mpfx::IEEE754Context ctx(f.es, f.nbits, rm);
    const size_t arity = (op == Op::SQRT) ? 1 : (op == Op::FMA) ? 3 : 2;
    const size_t n = vals.size();
    const size_t total = arity == 1 ? n : arity == 2 ? n * n : n * n * n;
    size_t reported = 0;

    cases += total;
    for (size_t idx = 0; idx < total; idx++) {
        const double x = vals[idx % n];
        const double y = arity >= 2 ? vals[(idx / n) % n] : 0.0;
        const double z = arity == 3 ? vals[idx / (n * n)] : 0.0;

        const double ref = oracle.eval(op, x, y, z, f, rm);
        const T got = run_op<E, T>(op, T(x), T(y), T(z), ctx);
        if (!same_value(got, T(ref))) {
            mismatches++;
            if (reported++ < 3) {
                ADD_FAILURE() << f.name() << " " << op_name(op) << " " << rm_name(rm)
                              << " [" << engine_name(E) << "/" << (sizeof(T) == 8 ? "f64" : "f32") << "]"
                              << " x=" << x << " y=" << y << " z=" << z
                              << " expected " << ref << " got " << double(got);
            }
        }
    }
}

template <std::floating_point T>
void check_all_engines(Op op, const Format& f, mpfx::RM rm, const std::vector<double>& vals,
                       Oracle& oracle, size_t& mismatches, size_t& cases) {
    check<mpfx::Engine::FP_RTO, T>(op, f, rm, vals, oracle, mismatches, cases);
    check<mpfx::Engine::EFT, T>(op, f, rm, vals, oracle, mismatches, cases);
    check<mpfx::Engine::SOFTFLOAT, T>(op, f, rm, vals, oracle, mismatches, cases);
    check<mpfx::Engine::FFLOAT, T>(op, f, rm, vals, oracle, mismatches, cases);
}

void run_exhaustive(const std::vector<Format>& formats, const std::vector<Op>& ops) {
    Oracle oracle;
    size_t cases = 0;
    for (const auto& f : formats) {
        const auto vals = enumerate(f);
        for (Op op : ops) {
            for (mpfx::RM rm : ALL_MODES) {
                size_t mismatches = 0;
                check_all_engines<double>(op, f, rm, vals, oracle, mismatches, cases);
                check_all_engines<float>(op, f, rm, vals, oracle, mismatches, cases);
                EXPECT_EQ(mismatches, 0u) << f.name() << " " << op_name(op) << " " << rm_name(rm);
            }
        }
    }
    // (engine, container, format, op, mode, input tuple) combinations checked
    std::cout << "[   INFO   ] " << formats.size() << " formats, " << cases << " cases\n";
}

// Every format with 2 <= es <= 5, p >= 2, and at most `max_bits` total bits.
std::vector<Format> formats_up_to(mpfx::prec_t max_bits) {
    std::vector<Format> out;
    for (mpfx::prec_t es = 2; es <= 5; es++) {
        for (mpfx::prec_t nbits = es + 2; nbits <= max_bits; nbits++) {
            out.push_back({es, nbits});
        }
    }
    return out;
}

} // anonymous namespace

TEST(Exhaustive, BinaryOpsUpTo8Bits) {
    run_exhaustive(formats_up_to(8), BINARY_OPS);
}

TEST(Exhaustive, SqrtUpTo8Bits) {
    run_exhaustive(formats_up_to(8), {Op::SQRT});
}

TEST(Exhaustive, FmaUpTo7Bits) {
    run_exhaustive(formats_up_to(7), {Op::FMA});
}
