/**
 * @file benchmark_ops.cpp
 * @brief Benchmarks each MPFX engine against other number libraries for an
 *        FP32 target.
 *
 * Inputs are sampled in FP32. The baselines (native hardware, MPFR, SoftFloat,
 * FloppyFloat) compute directly in FP32 -- the natural fast path for the
 * format. MPFX instead emulates FP32 rounding in an FP64 container: round-to-odd
 * needs `prec + 2 = 26` guard bits, which do not fit in the 24-bit f32 format,
 * so the engines work in `double` and round to the FP32 output format.
 *
 * Emits one CSV row per operation with timings in microseconds, then a second
 * table of speedups relative to SoftFloat. A single run does no repetition;
 * drive repeated runs from an external harness and aggregate there. Usage:
 *
 *     benchmark_ops <num_inputs> <rounding_mode>
 *
 * where `rounding_mode` is one of rne|rtp|rtn|rtz|raz.
 */

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <mpfr.h>
#include <mpfx.hpp>
#include <floppy_float.h>

extern "C" {
    #include <softfloat.h>
}


enum class OP1 {
    SQRT
};

enum class OP2 {
    ADD,
    SUB,
    MUL,
    DIV
};

enum class OP3 {
    FMA
};


inline std::string to_string(OP1 op) {
    switch (op) {
        case OP1::SQRT:
            return "sqrt";
        default:
            MPFX_ASSERT(false, "unsupported OP1");
    }
}

inline std::string to_string(OP2 op) {
    switch (op) {
        case OP2::ADD:
            return "add";
        case OP2::SUB:
            return "sub";
        case OP2::MUL:
            return "mul";
        case OP2::DIV:
            return "div";
        default:
            MPFX_ASSERT(false, "unsupported OP2");
    }
}

inline std::string to_string(OP3 op) {
    switch (op) {
        case OP3::FMA:
            return "fma";
        default:
            MPFX_ASSERT(false, "unsupported OP3");
    }
}


// Samples `vals.size()` FP32 values uniformly from [lower, upper].
static void generate_inputs(std::vector<float>& vals, double lower = -1.0, double upper = 1.0) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(lower, upper);

    for (size_t i = 0; i < vals.size(); i++) {
        vals[i] = static_cast<float>(dist(rng));
    }
}

// One benchmarked operation: timings (microseconds) for every implementation.
struct Row {
    std::string op;
    double native;
    double mpfr;
    double softfloat;
    double floppyfloat;
    double mpfx_rto;
    double mpfx_softfloat;
    double mpfx_ffloat;
    double mpfx_eft;
};

static void print_header() {
    std::cout << "op"
        << ", native"
        << ", mpfr"
        << ", softfloat"
        << ", floppyfloat"
        << ", mpfx_rto"
        << ", mpfx_sfloat"
        << ", mpfx_ffloat"
        << ", mpfx_eft"
        << "\n";
}

// Raw runtimes, in microseconds.
static void print_runtime_row(const Row& r) {
    std::cout << r.op
        << ", " << static_cast<size_t>(r.native)
        << ", " << static_cast<size_t>(r.mpfr)
        << ", " << static_cast<size_t>(r.softfloat)
        << ", " << static_cast<size_t>(r.floppyfloat)
        << ", " << static_cast<size_t>(r.mpfx_rto)
        << ", " << static_cast<size_t>(r.mpfx_softfloat)
        << ", " << static_cast<size_t>(r.mpfx_ffloat)
        << ", " << static_cast<size_t>(r.mpfx_eft)
        << "\n";
}

// Speedup relative to SoftFloat: softfloat_time / column_time. A value > 1 means
// faster than SoftFloat, < 1 means slower. (SoftFloat's own column is 1.00.)
static void print_speedup_row(const Row& r) {
    const auto sp = [&](double t) {
        std::cout << ", ";
        if (t > 0.0) {
            std::cout << r.softfloat / t;
        } else {
            std::cout << "inf";
        }
    };
    std::cout << r.op;
    sp(r.native);
    sp(r.mpfr);
    sp(r.softfloat);
    sp(r.floppyfloat);
    sp(r.mpfx_rto);
    sp(r.mpfx_softfloat);
    sp(r.mpfx_ffloat);
    sp(r.mpfx_eft);
    std::cout << "\n";
}

///////////////////////////////////////////////////////////
// Native FP32 references.
//
// Note: hardware ops use the FPU's current rounding mode (round-to-nearest);
// this column is a raw-throughput reference and does not honor a non-RNE mode.

template <OP1 O>
double reference_op1(const std::vector<float>& x_vals, size_t N) {
    auto start = std::chrono::steady_clock::now();

    volatile float result = 0.0f;
    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP1::SQRT) {
            result = std::sqrt(x_vals[i]);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    return duration;
}

template <OP2 O>
double reference_op2(const std::vector<float>& x_vals, const std::vector<float>& y_vals, size_t N) {
    auto start = std::chrono::steady_clock::now();
    volatile float result = 0.0f;
    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP2::ADD) {
            result = x_vals[i] + y_vals[i];
        } else if constexpr (O == OP2::SUB) {
            result = x_vals[i] - y_vals[i];
        } else if constexpr (O == OP2::MUL) {
            result = x_vals[i] * y_vals[i];
        } else if constexpr (O == OP2::DIV) {
            result = x_vals[i] / y_vals[i];
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP2");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    return duration;
}

template <OP3 O>
double reference_op3(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const std::vector<float>& z_vals, size_t N) {
    auto start = std::chrono::steady_clock::now();
    volatile float result = 0.0f;
    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP3::FMA) {
            result = std::fma(x_vals[i], y_vals[i], z_vals[i]);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP3");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    return duration;
}

////////////////////////////////////////////////////////////
// MPFR references (operands and result at FP32 precision).

static inline mpfr_rnd_t cvt_rm(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE:
            return MPFR_RNDN;
        case mpfx::RM::RTP:
            return MPFR_RNDU;
        case mpfx::RM::RTN:
            return MPFR_RNDD;
        case mpfx::RM::RTZ:
            return MPFR_RNDZ;
        case mpfx::RM::RAZ:
            return MPFR_RNDA;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

template <OP1 O>
double mpfr_op1(const std::vector<float>& x_vals, const mpfx::Context& ctx, size_t N) {
    mpfr_t mx, mr;
    mpfr_init2(mx, 24);
    mpfr_init2(mr, ctx.prec());
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_flt(mx, x_vals[i], MPFR_RNDN);
        if constexpr (O == OP1::SQRT) {
            mpfr_sqrt(mr, mx, rnd);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
        result = mpfr_get_d(mr, MPFR_RNDN);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;

    mpfr_clear(mx);
    mpfr_clear(mr);

    return duration;
}

template <OP2 O>
double mpfr_op2(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const mpfx::Context& ctx, size_t N) {
    mpfr_t mx, my, mr;
    mpfr_init2(mx, 24);
    mpfr_init2(my, 24);
    mpfr_init2(mr, ctx.prec());
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_flt(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_flt(my, y_vals[i], MPFR_RNDN);
        if constexpr (O == OP2::ADD) {
            mpfr_add(mr, mx, my, rnd);
        } else if constexpr (O == OP2::SUB) {
            mpfr_sub(mr, mx, my, rnd);
        } else if constexpr (O == OP2::MUL) {
            mpfr_mul(mr, mx, my, rnd);
        } else if constexpr (O == OP2::DIV) {
            mpfr_div(mr, mx, my, rnd);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP2");
        }
        result = mpfr_get_d(mr, MPFR_RNDN);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);

    return duration;
}

template <OP3 O>
double mpfr_op3(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const std::vector<float>& z_vals, const mpfx::Context& ctx, size_t N) {
    mpfr_t mx, my, mz, mr;
    mpfr_init2(mx, 24);
    mpfr_init2(my, 24);
    mpfr_init2(mz, 24);
    mpfr_init2(mr, ctx.prec());
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_flt(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_flt(my, y_vals[i], MPFR_RNDN);
        mpfr_set_flt(mz, z_vals[i], MPFR_RNDN);
        if constexpr (O == OP3::FMA) {
            mpfr_fma(mr, mx, my, mz, rnd);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP3");
        }
        result = mpfr_get_d(mr, MPFR_RNDN);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mz);
    mpfr_clear(mr);

    return duration;
}

////////////////////////////////////////////////////////////
// SoftFloat references (FP32).

static inline uint8_t cvt_rm_softfloat(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE:
            return softfloat_round_near_even;
        case mpfx::RM::RTP:
            return softfloat_round_max;
        case mpfx::RM::RTN:
            return softfloat_round_min;
        case mpfx::RM::RTZ:
            return softfloat_round_minMag;
        case mpfx::RM::RAZ:
            return softfloat_round_near_maxMag;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

template <OP1 O>
double softfloat_op1(const std::vector<float>& x_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        float32_t x{std::bit_cast<uint32_t>(x_vals[i])};
        if constexpr (O == OP1::SQRT) {
            result = std::bit_cast<float>(f32_sqrt(x).v);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <OP2 O>
double softfloat_op2(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        float32_t x{std::bit_cast<uint32_t>(x_vals[i])};
        float32_t y{std::bit_cast<uint32_t>(y_vals[i])};
        float32_t r;
        if constexpr (O == OP2::ADD) {
            r = f32_add(x, y);
        } else if constexpr (O == OP2::SUB) {
            r = f32_sub(x, y);
        } else if constexpr (O == OP2::MUL) {
            r = f32_mul(x, y);
        } else if constexpr (O == OP2::DIV) {
            r = f32_div(x, y);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP2");
        }
        result = std::bit_cast<float>(r.v);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <OP3 O>
double softfloat_op3(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const std::vector<float>& z_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        float32_t x{std::bit_cast<uint32_t>(x_vals[i])};
        float32_t y{std::bit_cast<uint32_t>(y_vals[i])};
        float32_t z{std::bit_cast<uint32_t>(z_vals[i])};
        if constexpr (O == OP3::FMA) {
            result = std::bit_cast<float>(f32_mulAdd(x, y, z).v);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP3");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

////////////////////////////////////////////////////////////
// FloppyFloat references (FP32).

static inline Vfpu::RoundingMode cvt_rm_floppyfloat(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE:
            return Vfpu::kRoundTiesToEven;
        case mpfx::RM::RTP:
            return Vfpu::kRoundTowardPositive;
        case mpfx::RM::RTN:
            return Vfpu::kRoundTowardNegative;
        case mpfx::RM::RTZ:
            return Vfpu::kRoundTowardZero;
        case mpfx::RM::RAZ:
            return Vfpu::kRoundTiesToAway;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

template <OP1 O>
double floppyfloat_op1(const std::vector<float>& x_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP1::SQRT) {
            result = ff.Sqrt(x_vals[i]);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <OP2 O>
double floppyfloat_op2(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP2::ADD) {
            result = ff.Add(x_vals[i], y_vals[i]);
        } else if constexpr (O == OP2::SUB) {
            result = ff.Sub(x_vals[i], y_vals[i]);
        } else if constexpr (O == OP2::MUL) {
            result = ff.Mul(x_vals[i], y_vals[i]);
        } else if constexpr (O == OP2::DIV) {
            result = ff.Div(x_vals[i], y_vals[i]);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP2");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <OP3 O>
double floppyfloat_op3(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const std::vector<float>& z_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP3::FMA) {
            result = ff.Fma(x_vals[i], y_vals[i], z_vals[i]);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP3");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

////////////////////////////////////////////////////////////
// MPFX engine implementations. Inputs are FP32 values widened to `double`
// (exact); each engine emulates FP32 rounding in the f64 container.

template <mpfx::Engine E, OP1 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
double mpfx_op1(const std::vector<double>& x_vals, const mpfx::Context& ctx, size_t N) {
    volatile double result = 0.0;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP1::SQRT) {
            result = mpfx::sqrt<E, Flags>(x_vals[i], ctx);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <mpfx::Engine E, OP2 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
double mpfx_op2(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const mpfx::Context& ctx, size_t N) {
    volatile double result = 0.0;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP2::ADD) {
            result = mpfx::add<E, Flags>(x_vals[i], y_vals[i], ctx);
        } else if constexpr (O == OP2::SUB) {
            result = mpfx::sub<E, Flags>(x_vals[i], y_vals[i], ctx);
        } else if constexpr (O == OP2::MUL) {
            result = mpfx::mul<E, Flags>(x_vals[i], y_vals[i], ctx);
        } else if constexpr (O == OP2::DIV) {
            result = mpfx::div<E, Flags>(x_vals[i], y_vals[i], ctx);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP2");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <mpfx::Engine E, OP3 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
double mpfx_op3(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const std::vector<double>& z_vals, const mpfx::Context& ctx, size_t N) {
    volatile double result = 0.0;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP3::FMA) {
            result = mpfx::fma<E, Flags>(x_vals[i], y_vals[i], z_vals[i], ctx);
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP3");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

////////////////////////////////////////////////////////////
// Benchmarking functions

template <OP1 O>
Row benchmark_op1(const mpfx::Context& output_ctx, size_t N) {
    std::vector<float> x_vals(N);
    generate_inputs(x_vals, 0.0, 1.0); // sqrt requires non-negative inputs
    const std::vector<double> x_wide(x_vals.begin(), x_vals.end());

    return Row{
        to_string(O),
        reference_op1<O>(x_vals, N),
        mpfr_op1<O>(x_vals, output_ctx, N),
        softfloat_op1<O>(x_vals, output_ctx, N),
        floppyfloat_op1<O>(x_vals, output_ctx, N),
        mpfx_op1<mpfx::Engine::FP_RTO, O>(x_wide, output_ctx, N),
        mpfx_op1<mpfx::Engine::SOFTFLOAT, O>(x_wide, output_ctx, N),
        mpfx_op1<mpfx::Engine::FFLOAT, O>(x_wide, output_ctx, N),
        mpfx_op1<mpfx::Engine::EFT, O>(x_wide, output_ctx, N)
    };
}

template <OP2 O>
Row benchmark_op2(const mpfx::Context& output_ctx, size_t N) {
    std::vector<float> x_vals(N);
    std::vector<float> y_vals(N);
    generate_inputs(x_vals);
    generate_inputs(y_vals);
    const std::vector<double> x_wide(x_vals.begin(), x_vals.end());
    const std::vector<double> y_wide(y_vals.begin(), y_vals.end());

    return Row{
        to_string(O),
        reference_op2<O>(x_vals, y_vals, N),
        mpfr_op2<O>(x_vals, y_vals, output_ctx, N),
        softfloat_op2<O>(x_vals, y_vals, output_ctx, N),
        floppyfloat_op2<O>(x_vals, y_vals, output_ctx, N),
        mpfx_op2<mpfx::Engine::FP_RTO, O>(x_wide, y_wide, output_ctx, N),
        mpfx_op2<mpfx::Engine::SOFTFLOAT, O>(x_wide, y_wide, output_ctx, N),
        mpfx_op2<mpfx::Engine::FFLOAT, O>(x_wide, y_wide, output_ctx, N),
        mpfx_op2<mpfx::Engine::EFT, O>(x_wide, y_wide, output_ctx, N)
    };
}

template <OP3 O>
Row benchmark_op3(const mpfx::Context& output_ctx, size_t N) {
    std::vector<float> x_vals(N);
    std::vector<float> y_vals(N);
    std::vector<float> z_vals(N);
    generate_inputs(x_vals);
    generate_inputs(y_vals);
    generate_inputs(z_vals);
    const std::vector<double> x_wide(x_vals.begin(), x_vals.end());
    const std::vector<double> y_wide(y_vals.begin(), y_vals.end());
    const std::vector<double> z_wide(z_vals.begin(), z_vals.end());

    return Row{
        to_string(O),
        reference_op3<O>(x_vals, y_vals, z_vals, N),
        mpfr_op3<O>(x_vals, y_vals, z_vals, output_ctx, N),
        softfloat_op3<O>(x_vals, y_vals, z_vals, output_ctx, N),
        floppyfloat_op3<O>(x_vals, y_vals, z_vals, output_ctx, N),
        mpfx_op3<mpfx::Engine::FP_RTO, O>(x_wide, y_wide, z_wide, output_ctx, N),
        mpfx_op3<mpfx::Engine::SOFTFLOAT, O>(x_wide, y_wide, z_wide, output_ctx, N),
        mpfx_op3<mpfx::Engine::FFLOAT, O>(x_wide, y_wide, z_wide, output_ctx, N),
        mpfx_op3<mpfx::Engine::EFT, O>(x_wide, y_wide, z_wide, output_ctx, N)
    };
}

// Parses a rounding-mode name. Limited to the modes the reference converters
// (`cvt_rm`, SoftFloat, FloppyFloat) support.
static mpfx::RM parse_rm(const std::string& s) {
    if (s == "rne") return mpfx::RM::RNE;
    if (s == "rtp") return mpfx::RM::RTP;
    if (s == "rtn") return mpfx::RM::RTN;
    if (s == "rtz") return mpfx::RM::RTZ;
    if (s == "raz") return mpfx::RM::RAZ;
    std::cerr << "unknown rounding mode '" << s << "' (use rne|rtp|rtn|rtz|raz)\n";
    std::exit(1);
}

int main(int argc, char** argv) {
    // CLI: benchmark_ops <num_inputs> <rounding_mode>
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <num_inputs> <rounding_mode>\n"
                  << "  rounding_mode: rne|rtp|rtn|rtz|raz\n";
        return 1;
    }
    const size_t N = std::stoull(argv[1]);
    const mpfx::RM rm = parse_rm(argv[2]);

    // FP32 target: the engines round to a 24-bit IEEE-754 format (f64 container).
    const auto output_ctx = mpfx::IEEE754Context(8, 32, rm);

    const std::vector<Row> rows = {
        benchmark_op2<OP2::ADD>(output_ctx, N),
        benchmark_op2<OP2::SUB>(output_ctx, N),
        benchmark_op2<OP2::MUL>(output_ctx, N),
        benchmark_op2<OP2::DIV>(output_ctx, N),
        benchmark_op1<OP1::SQRT>(output_ctx, N),
        benchmark_op3<OP3::FMA>(output_ctx, N)
    };

    // Table 1: raw runtimes (microseconds).
    print_header();
    for (const auto& r : rows) print_runtime_row(r);

    // Table 2: speedup relative to SoftFloat (>1 = faster than SoftFloat).
    std::cout << "\n# speedup relative to softfloat (>1 = faster, <1 = slower)\n";
    std::cout << std::fixed << std::setprecision(2);
    print_header();
    for (const auto& r : rows) print_speedup_row(r);

    return 0;
}
