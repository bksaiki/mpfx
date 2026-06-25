/**
 * @file benchmark_ops.cpp
 * @brief Benchmarks each MPFX engine against other number libraries.
 *
 * For each container type (`f32` and `f64`) it emits one CSV row per operation
 * with timings in microseconds. A single run does no repetition; drive repeated
 * runs from an external harness and aggregate there. Usage:
 *
 *     benchmark_ops [num_inputs]
 *
 * Default: 10000000 inputs. The emulated output format is a fixed 16-bit
 * IEEE-754 format, small enough to leave guard bits in the f32 container.
 */

#include <bit>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <concepts>
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


template <std::floating_point T>
static void generate_inputs(std::vector<T>& vals, const mpfx::Context& ctx, double lower = -1.0, double upper = 1.0) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(lower, upper);

    for (size_t i = 0; i < vals.size(); i++) {
        vals[i] = static_cast<T>(mpfx::round(dist(rng), ctx));
    }
}

// One benchmarked operation: timings (microseconds) for every implementation.
struct Row {
    std::string type;
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
    std::cout << "type"
        << ", op"
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
    std::cout << r.type
        << ", " << r.op
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
    std::cout << r.type << ", " << r.op;
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
// Reference implementations (native hardware arithmetic in the container type)

template <std::floating_point T, OP1 O>
double reference_op1(const std::vector<T>& x_vals, size_t N) {
    auto start = std::chrono::steady_clock::now();

    volatile T result = static_cast<T>(0);
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

template <std::floating_point T, OP2 O>
double reference_op2(const std::vector<T>& x_vals, const std::vector<T>& y_vals, size_t N) {
    auto start = std::chrono::steady_clock::now();
    volatile T result = static_cast<T>(0);
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

template <std::floating_point T, OP3 O>
double reference_op3(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const std::vector<T>& z_vals, size_t N) {
    auto start = std::chrono::steady_clock::now();
    volatile T result = static_cast<T>(0);
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
// MPFR implementations

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

// Precision of the container type, used for MPFR input operands.
template <std::floating_point T>
static constexpr mpfr_prec_t container_prec() {
    return std::is_same_v<T, float> ? 24 : 53;
}

// Set an MPFR operand from a container-type value (exactly).
template <std::floating_point T>
static inline void mpfr_set_t(mpfr_t dst, T value) {
    if constexpr (std::is_same_v<T, float>) {
        mpfr_set_flt(dst, value, MPFR_RNDN);
    } else {
        mpfr_set_d(dst, value, MPFR_RNDN);
    }
}

template <std::floating_point T, OP1 O>
double mpfr_op1(const std::vector<T>& x_vals, const mpfx::Context& ctx, size_t N) {
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    mpfr_t mx, mr;
    mpfr_init2(mx, container_prec<T>());
    mpfr_init2(mr, ctx.prec());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_t(mx, x_vals[i]);
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

template <std::floating_point T, OP2 O>
double mpfr_op2(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const mpfx::Context& ctx, size_t N) {
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    mpfr_t mx, my, mr;
    mpfr_init2(mx, container_prec<T>());
    mpfr_init2(my, container_prec<T>());
    mpfr_init2(mr, ctx.prec());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_t(mx, x_vals[i]);
        mpfr_set_t(my, y_vals[i]);
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

template <std::floating_point T, OP3 O>
double mpfr_op3(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const std::vector<T>& z_vals, const mpfx::Context& ctx, size_t N) {
    const mpfr_rnd_t rnd = cvt_rm(ctx.rm());

    mpfr_t mx, my, mz, mr;
    mpfr_init2(mx, container_prec<T>());
    mpfr_init2(my, container_prec<T>());
    mpfr_init2(mz, container_prec<T>());
    mpfr_init2(mr, ctx.prec());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_t(mx, x_vals[i]);
        mpfr_set_t(my, y_vals[i]);
        mpfr_set_t(mz, z_vals[i]);
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
// SoftFloat implementations

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

template <std::floating_point T, OP1 O>
double softfloat_op1(const std::vector<T>& x_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile T result = static_cast<T>(0);
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP1::SQRT) {
            if constexpr (std::is_same_v<T, float>) {
                float32_t x{std::bit_cast<uint32_t>(x_vals[i])};
                result = std::bit_cast<float>(f32_sqrt(x).v);
            } else {
                float64_t x{std::bit_cast<uint64_t>(x_vals[i])};
                result = std::bit_cast<double>(f64_sqrt(x).v);
            }
        } else {
            MPFX_STATIC_ASSERT(false, "unsupported OP1");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <std::floating_point T, OP2 O>
double softfloat_op2(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile T result = static_cast<T>(0);
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (std::is_same_v<T, float>) {
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
        } else {
            float64_t x{std::bit_cast<uint64_t>(x_vals[i])};
            float64_t y{std::bit_cast<uint64_t>(y_vals[i])};
            float64_t r;
            if constexpr (O == OP2::ADD) {
                r = f64_add(x, y);
            } else if constexpr (O == OP2::SUB) {
                r = f64_sub(x, y);
            } else if constexpr (O == OP2::MUL) {
                r = f64_mul(x, y);
            } else if constexpr (O == OP2::DIV) {
                r = f64_div(x, y);
            } else {
                MPFX_STATIC_ASSERT(false, "unsupported OP2");
            }
            result = std::bit_cast<double>(r.v);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    (void) result;
    return duration;
}

template <std::floating_point T, OP3 O>
double softfloat_op3(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const std::vector<T>& z_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    volatile T result = static_cast<T>(0);
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP3::FMA) {
            if constexpr (std::is_same_v<T, float>) {
                float32_t x{std::bit_cast<uint32_t>(x_vals[i])};
                float32_t y{std::bit_cast<uint32_t>(y_vals[i])};
                float32_t z{std::bit_cast<uint32_t>(z_vals[i])};
                result = std::bit_cast<float>(f32_mulAdd(x, y, z).v);
            } else {
                float64_t x{std::bit_cast<uint64_t>(x_vals[i])};
                float64_t y{std::bit_cast<uint64_t>(y_vals[i])};
                float64_t z{std::bit_cast<uint64_t>(z_vals[i])};
                result = std::bit_cast<double>(f64_mulAdd(x, y, z).v);
            }
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
// FloppyFloat implementations

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

template <std::floating_point T, OP1 O>
double floppyfloat_op1(const std::vector<T>& x_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    volatile T result = static_cast<T>(0);
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

template <std::floating_point T, OP2 O>
double floppyfloat_op2(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    volatile T result = static_cast<T>(0);
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

template <std::floating_point T, OP3 O>
double floppyfloat_op3(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const std::vector<T>& z_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    volatile T result = static_cast<T>(0);
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
// MPFX engine implementations

template <mpfx::Engine E, std::floating_point T, OP1 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
double mpfx_op1(const std::vector<T>& x_vals, const mpfx::Context& ctx, size_t N) {
    volatile T result = static_cast<T>(0);
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

template <mpfx::Engine E, std::floating_point T, OP2 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
double mpfx_op2(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const mpfx::Context& ctx, size_t N) {
    volatile T result = static_cast<T>(0);
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

template <mpfx::Engine E, std::floating_point T, OP3 O, mpfx::flag_mask_t Flags = mpfx::Flags::ALL_FLAGS>
double mpfx_op3(const std::vector<T>& x_vals, const std::vector<T>& y_vals, const std::vector<T>& z_vals, const mpfx::Context& ctx, size_t N) {
    volatile T result = static_cast<T>(0);
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

template <std::floating_point T, OP1 O>
Row benchmark_op1(
    const std::string& type,
    const mpfx::Context& input_ctx,
    const mpfx::Context& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<T> x_vals(num_inputs);
    if constexpr (O == OP1::SQRT) {
        generate_inputs(x_vals, input_ctx, 0.0, 1.0); // sqrt requires non-negative inputs
    } else {
        generate_inputs(x_vals, input_ctx);
    }

    return Row{
        type, to_string(O),
        reference_op1<T, O>(x_vals, num_inputs),
        mpfr_op1<T, O>(x_vals, output_ctx, num_inputs),
        softfloat_op1<T, O>(x_vals, output_ctx, num_inputs),
        floppyfloat_op1<T, O>(x_vals, output_ctx, num_inputs),
        mpfx_op1<mpfx::Engine::FP_RTO, T, O>(x_vals, output_ctx, num_inputs),
        mpfx_op1<mpfx::Engine::SOFTFLOAT, T, O>(x_vals, output_ctx, num_inputs),
        mpfx_op1<mpfx::Engine::FFLOAT, T, O>(x_vals, output_ctx, num_inputs),
        mpfx_op1<mpfx::Engine::EFT, T, O>(x_vals, output_ctx, num_inputs)
    };
}

template <std::floating_point T, OP2 O>
Row benchmark_op2(
    const std::string& type,
    const mpfx::Context& input_ctx,
    const mpfx::Context& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<T> x_vals(num_inputs);
    std::vector<T> y_vals(num_inputs);
    generate_inputs(x_vals, input_ctx);
    generate_inputs(y_vals, input_ctx);

    return Row{
        type, to_string(O),
        reference_op2<T, O>(x_vals, y_vals, num_inputs),
        mpfr_op2<T, O>(x_vals, y_vals, output_ctx, num_inputs),
        softfloat_op2<T, O>(x_vals, y_vals, output_ctx, num_inputs),
        floppyfloat_op2<T, O>(x_vals, y_vals, output_ctx, num_inputs),
        mpfx_op2<mpfx::Engine::FP_RTO, T, O>(x_vals, y_vals, output_ctx, num_inputs),
        mpfx_op2<mpfx::Engine::SOFTFLOAT, T, O>(x_vals, y_vals, output_ctx, num_inputs),
        mpfx_op2<mpfx::Engine::FFLOAT, T, O>(x_vals, y_vals, output_ctx, num_inputs),
        mpfx_op2<mpfx::Engine::EFT, T, O>(x_vals, y_vals, output_ctx, num_inputs)
    };
}

template <std::floating_point T, OP3 O>
Row benchmark_op3(
    const std::string& type,
    const mpfx::Context& input_ctx,
    const mpfx::Context& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<T> x_vals(num_inputs);
    std::vector<T> y_vals(num_inputs);
    std::vector<T> z_vals(num_inputs);
    generate_inputs(x_vals, input_ctx);
    generate_inputs(y_vals, input_ctx);
    generate_inputs(z_vals, input_ctx);

    return Row{
        type, to_string(O),
        reference_op3<T, O>(x_vals, y_vals, z_vals, num_inputs),
        mpfr_op3<T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs),
        softfloat_op3<T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs),
        floppyfloat_op3<T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs),
        mpfx_op3<mpfx::Engine::FP_RTO, T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs),
        mpfx_op3<mpfx::Engine::SOFTFLOAT, T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs),
        mpfx_op3<mpfx::Engine::FFLOAT, T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs),
        mpfx_op3<mpfx::Engine::EFT, T, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs)
    };
}

// Emulated output format: a fixed 16-bit IEEE-754 format (8 exponent bits),
// small enough that round_prec (= prec + 2 = 10) leaves guard bits in both the
// f32 (24-bit) and f64 (53-bit) containers.
static const auto OUTPUT_CTX = mpfx::IEEE754Context(8, 16, mpfx::RM::RNE);

// Runs the full operation suite in a given container type, appending one row
// per operation to `rows`.
template <std::floating_point T>
static void run(std::vector<Row>& rows, const std::string& type, const mpfx::Context& input_ctx, size_t N) {
    rows.push_back(benchmark_op2<T, OP2::ADD>(type, input_ctx, OUTPUT_CTX, N));
    rows.push_back(benchmark_op2<T, OP2::SUB>(type, input_ctx, OUTPUT_CTX, N));
    rows.push_back(benchmark_op2<T, OP2::MUL>(type, input_ctx, OUTPUT_CTX, N));
    rows.push_back(benchmark_op2<T, OP2::DIV>(type, input_ctx, OUTPUT_CTX, N));
    rows.push_back(benchmark_op1<T, OP1::SQRT>(type, input_ctx, OUTPUT_CTX, N));
    rows.push_back(benchmark_op3<T, OP3::FMA>(type, input_ctx, OUTPUT_CTX, N));
}

int main(int argc, char** argv) {
    // CLI: benchmark_ops [num_inputs]
    size_t N = 10'000'000;
    if (argc > 1) N = std::stoull(argv[1]);

    std::vector<Row> rows;
    // f32 container: inputs are FP32; f64 container: inputs are FP64.
    run<float>(rows, "f32", mpfx::IEEE754Context(8, 32, mpfx::RM::RNE), N);
    run<double>(rows, "f64", mpfx::IEEE754Context(11, 64, mpfx::RM::RNE), N);

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
