/**
 * @file benchmark_ops.cpp
 * @brief Benchmarks each MPFX engine against other number libraries for a
 *        uniform-precision FP32 or FP16 target.
 *
 * Inputs are sampled as exact values in the target format. The baselines (MPFR,
 * SoftFloat, FloppyFloat) compute directly in the target format. MPFX instead
 * emulates the format's rounding in an FP64 container: round-to-odd needs
 * `prec + 2` guard bits, which do not fit in the narrow format, so the engines
 * work in `double` and round to the target output format.
 *
 * The target format is selected on the command line. FloppyFloat only provides
 * f32/f64 instantiations, so its column is reported as `n/a` for FP16. The
 * native-hardware reference is intentionally omitted: there is no portable
 * half-precision arithmetic type (Clang lacks `_Float16` arithmetic on x86),
 * and FP16 inputs/bit patterns are produced via SoftFloat conversions instead.
 *
 * Emits one CSV row per operation with timings in microseconds, then a second
 * table of speedups relative to SoftFloat. A single run does no repetition;
 * drive repeated runs from an external harness and aggregate there. Usage:
 *
 *     benchmark_ops <num_inputs> <rounding_mode> [format]
 *
 * where `rounding_mode` is one of rne|rtp|rtn|rtz|raz and `format` is one of
 * fp16|fp32 (default fp32).
 */

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
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


// SoftFloat type used to represent inputs/results for a given target format.
template <bool FP16>
using sf_t = std::conditional_t<FP16, float16_t, float32_t>;

// Samples `wide.size()` values uniformly from [lower, upper], rounded to the
// target format. Each input is produced both as an exact `double` (for the MPFR
// and MPFX paths) and as the matching SoftFloat type (for the SoftFloat path).
template <bool FP16>
static void generate_inputs(std::vector<double>& wide, std::vector<sf_t<FP16>>& sf,
                            double lower = -1.0, double upper = 1.0) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(lower, upper);
    softfloat_roundingMode = softfloat_round_near_even;

    for (size_t i = 0; i < wide.size(); i++) {
        const double d0 = dist(rng);
        if constexpr (FP16) {
            // round the sample into FP16, then widen back exactly to double
            const float16_t h = f64_to_f16(std::bit_cast<float64_t>(d0));
            sf[i] = h;
            wide[i] = std::bit_cast<double>(f16_to_f64(h));
        } else {
            // round the sample into FP32, then widen exactly to double
            const float f = static_cast<float>(d0);
            sf[i] = std::bit_cast<float32_t>(f);
            wide[i] = static_cast<double>(f);
        }
    }
}

// One benchmarked operation: timings (microseconds) for every implementation.
// `floppyfloat < 0` marks a column that was not run (e.g. FP16, unsupported).
struct Row {
    std::string op;
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
        << ", mpfr"
        << ", softfloat"
        << ", floppyfloat"
        << ", mpfx_rto"
        << ", mpfx_sfloat"
        << ", mpfx_ffloat"
        << ", mpfx_eft"
        << "\n";
}

// Raw runtimes, in microseconds. A negative timing prints as `n/a`.
static void print_runtime_row(const Row& r) {
    const auto t = [](double v) {
        if (v < 0.0) {
            std::cout << "n/a";
        } else {
            std::cout << static_cast<size_t>(v);
        }
    };
    std::cout << r.op;
    std::cout << ", "; t(r.mpfr);
    std::cout << ", "; t(r.softfloat);
    std::cout << ", "; t(r.floppyfloat);
    std::cout << ", "; t(r.mpfx_rto);
    std::cout << ", "; t(r.mpfx_softfloat);
    std::cout << ", "; t(r.mpfx_ffloat);
    std::cout << ", "; t(r.mpfx_eft);
    std::cout << "\n";
}

// Speedup relative to SoftFloat: softfloat_time / column_time. A value > 1 means
// faster than SoftFloat, < 1 means slower. (SoftFloat's own column is 1.00.)
// A negative timing (not run) prints as `n/a`.
static void print_speedup_row(const Row& r) {
    const auto sp = [&](double t) {
        std::cout << ", ";
        if (t < 0.0) {
            std::cout << "n/a";
        } else if (t > 0.0) {
            std::cout << r.softfloat / t;
        } else {
            std::cout << "inf";
        }
    };
    std::cout << r.op;
    sp(r.mpfr);
    sp(r.softfloat);
    sp(r.floppyfloat);
    sp(r.mpfx_rto);
    sp(r.mpfx_softfloat);
    sp(r.mpfx_ffloat);
    sp(r.mpfx_eft);
    std::cout << "\n";
}

////////////////////////////////////////////////////////////
// MPFR references (operands and result at the target precision).

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
double mpfr_op1(const std::vector<double>& x_vals, const mpfx::Context& ctx, size_t N) {
    mpfr_t mx, mr;
    mpfr_init2(mx, ctx.prec());
    mpfr_init2(mr, ctx.prec());
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
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
double mpfr_op2(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const mpfx::Context& ctx, size_t N) {
    mpfr_t mx, my, mr;
    mpfr_init2(mx, ctx.prec());
    mpfr_init2(my, ctx.prec());
    mpfr_init2(mr, ctx.prec());
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_d(my, y_vals[i], MPFR_RNDN);
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
double mpfr_op3(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const std::vector<double>& z_vals, const mpfx::Context& ctx, size_t N) {
    mpfr_t mx, my, mz, mr;
    mpfr_init2(mx, ctx.prec());
    mpfr_init2(my, ctx.prec());
    mpfr_init2(mz, ctx.prec());
    mpfr_init2(mr, ctx.prec());
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_d(my, y_vals[i], MPFR_RNDN);
        mpfr_set_d(mz, z_vals[i], MPFR_RNDN);
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
// SoftFloat references (target format).

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

template <bool FP16, OP1 O>
double softfloat_op1(const std::vector<sf_t<FP16>>& x_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile uint64_t sink = 0;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        sf_t<FP16> r;
        if constexpr (O == OP1::SQRT) {
            if constexpr (FP16) {
                r = f16_sqrt(x_vals[i]);
            } else {
                r = f32_sqrt(x_vals[i]);
            }
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
        sink = r.v;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) sink;
    return duration;
}

template <bool FP16, OP2 O>
double softfloat_op2(const std::vector<sf_t<FP16>>& x_vals, const std::vector<sf_t<FP16>>& y_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile uint64_t sink = 0;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        sf_t<FP16> r;
        if constexpr (FP16) {
            if constexpr (O == OP2::ADD) {
                r = f16_add(x_vals[i], y_vals[i]);
            } else if constexpr (O == OP2::SUB) {
                r = f16_sub(x_vals[i], y_vals[i]);
            } else if constexpr (O == OP2::MUL) {
                r = f16_mul(x_vals[i], y_vals[i]);
            } else if constexpr (O == OP2::DIV) {
                r = f16_div(x_vals[i], y_vals[i]);
            } else {
                MPFX_STATIC_ASSERT(false, "unsupported OP2");
            }
        } else {
            if constexpr (O == OP2::ADD) {
                r = f32_add(x_vals[i], y_vals[i]);
            } else if constexpr (O == OP2::SUB) {
                r = f32_sub(x_vals[i], y_vals[i]);
            } else if constexpr (O == OP2::MUL) {
                r = f32_mul(x_vals[i], y_vals[i]);
            } else if constexpr (O == OP2::DIV) {
                r = f32_div(x_vals[i], y_vals[i]);
            } else {
                MPFX_STATIC_ASSERT(false, "unsupported OP2");
            }
        }
        sink = r.v;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) sink;
    return duration;
}

template <bool FP16, OP3 O>
double softfloat_op3(const std::vector<sf_t<FP16>>& x_vals, const std::vector<sf_t<FP16>>& y_vals, const std::vector<sf_t<FP16>>& z_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile uint64_t sink = 0;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        sf_t<FP16> r;
        if constexpr (O == OP3::FMA) {
            if constexpr (FP16) {
                r = f16_mulAdd(x_vals[i], y_vals[i], z_vals[i]);
            } else {
                r = f32_mulAdd(x_vals[i], y_vals[i], z_vals[i]);
            }
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP3");
        }
        sink = r.v;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) sink;
    return duration;
}

////////////////////////////////////////////////////////////
// FloppyFloat references. FloppyFloat only provides f32/f64 instantiations, so
// the FP16 path is not instantiated and reports `n/a` (negative timing).

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

template <bool FP16, OP1 O>
double floppyfloat_op1(const std::vector<float>& x_vals, const mpfx::Context& ctx, size_t N) {
    if constexpr (FP16) {
        (void) x_vals; (void) ctx; (void) N;
        return -1.0; // FloppyFloat has no FP16 instantiation
    } else {
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
}

template <bool FP16, OP2 O>
double floppyfloat_op2(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const mpfx::Context& ctx, size_t N) {
    if constexpr (FP16) {
        (void) x_vals; (void) y_vals; (void) ctx; (void) N;
        return -1.0; // FloppyFloat has no FP16 instantiation
    } else {
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
}

template <bool FP16, OP3 O>
double floppyfloat_op3(const std::vector<float>& x_vals, const std::vector<float>& y_vals, const std::vector<float>& z_vals, const mpfx::Context& ctx, size_t N) {
    if constexpr (FP16) {
        (void) x_vals; (void) y_vals; (void) z_vals; (void) ctx; (void) N;
        return -1.0; // FloppyFloat has no FP16 instantiation
    } else {
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
}

////////////////////////////////////////////////////////////
// MPFX engine implementations. Inputs are target-format values widened to
// `double` (exact); each engine emulates the format's rounding in the f64
// container.

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

template <bool FP16, OP1 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
Row benchmark_op1(const mpfx::Context& output_ctx, size_t N) {
    std::vector<double> x_wide(N);
    std::vector<sf_t<FP16>> x_sf(N);
    generate_inputs<FP16>(x_wide, x_sf, 0.0, 1.0); // sqrt requires non-negative inputs
    const std::vector<float> x_f(x_wide.begin(), x_wide.end());

    return Row{
        to_string(O),
        mpfr_op1<O>(x_wide, output_ctx, N),
        softfloat_op1<FP16, O>(x_sf, output_ctx, N),
        floppyfloat_op1<FP16, O>(x_f, output_ctx, N),
        mpfx_op1<mpfx::Engine::FP_RTO, O, Flags>(x_wide, output_ctx, N),
        mpfx_op1<mpfx::Engine::SOFTFLOAT, O, Flags>(x_wide, output_ctx, N),
        mpfx_op1<mpfx::Engine::FFLOAT, O, Flags>(x_wide, output_ctx, N),
        mpfx_op1<mpfx::Engine::EFT, O, Flags>(x_wide, output_ctx, N)
    };
}

template <bool FP16, OP2 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
Row benchmark_op2(const mpfx::Context& output_ctx, size_t N) {
    std::vector<double> x_wide(N), y_wide(N);
    std::vector<sf_t<FP16>> x_sf(N), y_sf(N);
    generate_inputs<FP16>(x_wide, x_sf);
    generate_inputs<FP16>(y_wide, y_sf);
    const std::vector<float> x_f(x_wide.begin(), x_wide.end());
    const std::vector<float> y_f(y_wide.begin(), y_wide.end());

    return Row{
        to_string(O),
        mpfr_op2<O>(x_wide, y_wide, output_ctx, N),
        softfloat_op2<FP16, O>(x_sf, y_sf, output_ctx, N),
        floppyfloat_op2<FP16, O>(x_f, y_f, output_ctx, N),
        mpfx_op2<mpfx::Engine::FP_RTO, O, Flags>(x_wide, y_wide, output_ctx, N),
        mpfx_op2<mpfx::Engine::SOFTFLOAT, O, Flags>(x_wide, y_wide, output_ctx, N),
        mpfx_op2<mpfx::Engine::FFLOAT, O, Flags>(x_wide, y_wide, output_ctx, N),
        mpfx_op2<mpfx::Engine::EFT, O, Flags>(x_wide, y_wide, output_ctx, N)
    };
}

template <bool FP16, OP3 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
Row benchmark_op3(const mpfx::Context& output_ctx, size_t N) {
    std::vector<double> x_wide(N), y_wide(N), z_wide(N);
    std::vector<sf_t<FP16>> x_sf(N), y_sf(N), z_sf(N);
    generate_inputs<FP16>(x_wide, x_sf);
    generate_inputs<FP16>(y_wide, y_sf);
    generate_inputs<FP16>(z_wide, z_sf);
    const std::vector<float> x_f(x_wide.begin(), x_wide.end());
    const std::vector<float> y_f(y_wide.begin(), y_wide.end());
    const std::vector<float> z_f(z_wide.begin(), z_wide.end());

    return Row{
        to_string(O),
        mpfr_op3<O>(x_wide, y_wide, z_wide, output_ctx, N),
        softfloat_op3<FP16, O>(x_sf, y_sf, z_sf, output_ctx, N),
        floppyfloat_op3<FP16, O>(x_f, y_f, z_f, output_ctx, N),
        mpfx_op3<mpfx::Engine::FP_RTO, O, Flags>(x_wide, y_wide, z_wide, output_ctx, N),
        mpfx_op3<mpfx::Engine::SOFTFLOAT, O, Flags>(x_wide, y_wide, z_wide, output_ctx, N),
        mpfx_op3<mpfx::Engine::FFLOAT, O, Flags>(x_wide, y_wide, z_wide, output_ctx, N),
        mpfx_op3<mpfx::Engine::EFT, O, Flags>(x_wide, y_wide, z_wide, output_ctx, N)
    };
}

// Runs the full operation suite for a target format selected by `FP16`.
template <bool FP16>
static void run(mpfx::RM rm, size_t N) {
    // Target format: FP16 is a 16-bit IEEE-754 format (5 exponent bits), FP32 a
    // 32-bit one (8 exponent bits). MPFX rounds to it from an f64 container.
    const auto output_ctx = FP16 ? mpfx::IEEE754Context(5, 16, rm)
                                 : mpfx::IEEE754Context(8, 32, rm);

    // Status-flag mask applied to every MPFX rounding in this run. Set to
    // `mpfx::Flags::NO_FLAGS` to measure rounding without flag bookkeeping, or
    // a specific subset (e.g. `mpfx::Flags::INEXACT_FLAG`) to isolate its cost.
    constexpr mpfx::flag_mask_t MPFX_FLAGS = mpfx::Flags::ALL_FLAGS;

    const std::vector<Row> rows = {
        benchmark_op2<FP16, OP2::ADD, MPFX_FLAGS>(output_ctx, N),
        benchmark_op2<FP16, OP2::SUB, MPFX_FLAGS>(output_ctx, N),
        benchmark_op2<FP16, OP2::MUL, MPFX_FLAGS>(output_ctx, N),
        benchmark_op2<FP16, OP2::DIV, MPFX_FLAGS>(output_ctx, N),
        benchmark_op1<FP16, OP1::SQRT, MPFX_FLAGS>(output_ctx, N),
        benchmark_op3<FP16, OP3::FMA, MPFX_FLAGS>(output_ctx, N)
    };

    // Print Config
    std::cout << "type: " << (FP16 ? "fp16" : "fp32") << "\n";

    // Table 1: raw runtimes (microseconds).
    print_header();
    for (const auto& r : rows) print_runtime_row(r);

    // Table 2: speedup relative to SoftFloat (>1 = faster than SoftFloat).
    std::cout << "\n# speedup relative to softfloat (>1 = faster, <1 = slower)\n";
    std::cout << std::fixed << std::setprecision(2);
    print_header();
    for (const auto& r : rows) print_speedup_row(r);
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
    // CLI: benchmark_ops <num_inputs> <rounding_mode> [format]
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: " << argv[0] << " <num_inputs> <rounding_mode> [format]\n"
                  << "  rounding_mode: rne|rtp|rtn|rtz|raz\n"
                  << "  format:        fp16|fp32 (default fp32)\n";
        return 1;
    }
    const size_t N = std::stoull(argv[1]);
    const mpfx::RM rm = parse_rm(argv[2]);
    const std::string fmt = (argc == 4) ? argv[3] : "fp32";

    if (fmt == "fp32") {
        run<false>(rm, N);
    } else if (fmt == "fp16") {
        run<true>(rm, N);
    } else {
        std::cerr << "unknown format '" << fmt << "' (use fp16|fp32)\n";
        return 1;
    }

    return 0;
}
