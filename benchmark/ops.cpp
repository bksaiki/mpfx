/**
 * @file ops.cpp
 * @brief Benchmarking against other number libraries.
 */

#include <bit>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <concepts>
#include <random>
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

static void report_header() {
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

static void report_result(
    const std::string& op_name,
    double duration_ref,
    double duration_mpfr,
    double duration_softfloat,
    double duration_floppyfloat,
    double duration_mpfx_rto,
    double duration_mpfx_softfloat,
    double duration_mpfx_ffloat,
    double duration_mpfx_eft
) {
    std::cout << op_name
        << ", " << static_cast<size_t>(duration_ref)
        << ", " << static_cast<size_t>(duration_mpfr)
        << ", " << static_cast<size_t>(duration_softfloat)
        << ", " << static_cast<size_t>(duration_floppyfloat)
        << ", " << static_cast<size_t>(duration_mpfx_rto)
        << ", " << static_cast<size_t>(duration_mpfx_softfloat)
        << ", " << static_cast<size_t>(duration_mpfx_ffloat)
        << ", " << static_cast<size_t>(duration_mpfx_eft)
        << "\n";
}

///////////////////////////////////////////////////////////
// Reference implementations

template <OP1 O>
double reference_op1(const std::vector<double>& x_vals, size_t N) {
    // Quantize to FP32
    std::vector<float> x_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
    }

    // Time and compute
    auto start = std::chrono::steady_clock::now();

    volatile float result = 0.0f;
    for (size_t i = 0; i < x_vals.size(); i++) {
        if constexpr (O == OP1::SQRT) {
            result = std::sqrt(x_fl[i]);
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
double reference_op2(const std::vector<double>& x_vals, const std::vector<double>& y_vals, size_t N) {
    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    // Time and compute
    auto start = std::chrono::steady_clock::now();
    volatile float result = 0.0f;
    for (size_t i = 0; i < x_vals.size(); i++) {
        if constexpr (O == OP2::ADD) {
            result = x_fl[i] + y_fl[i];
        } else if constexpr (O == OP2::SUB) {
            result = x_fl[i] - y_fl[i];
        } else if constexpr (O == OP2::MUL) {
            result = x_fl[i] * y_fl[i];
        } else if constexpr (O == OP2::DIV) {
            result = x_fl[i] / y_fl[i];
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
double reference_op3(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const std::vector<double>& z_vals, size_t N) {
    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    std::vector<float> z_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
        z_fl[i] = static_cast<float>(z_vals[i]);
    }

    // Time and compute
    auto start = std::chrono::steady_clock::now();
    volatile float result = 0.0f;
    for (size_t i = 0; i < x_vals.size(); i++) {
        if constexpr (O == OP3::FMA) {
            result = std::fma(x_fl[i], y_fl[i], z_fl[i]);
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

static mpfr_rnd_t cvt_rm(mpfx::RM rm) {
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
    // Quantize to FP32
    std::vector<float> x_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
    }

    mpfr_t mx, mr;
    mpfr_init2(mx, 24);
    mpfr_init2(mr, ctx.prec());
    
    volatile double result = 0.0;
    
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        mpfr_set_flt(mx, x_fl[i], MPFR_RNDN);
        if constexpr (O == OP1::SQRT) {
            mpfr_sqrt(mr, mx, cvt_rm(ctx.rm()));
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
    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    mpfr_t mx, my, mr;
    mpfr_init2(mx, 24);
    mpfr_init2(my, 24);
    mpfr_init2(mr, ctx.prec());
    
    volatile double result = 0.0;
    
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        mpfr_set_flt(mx, x_fl[i], MPFR_RNDN);
        mpfr_set_flt(my, y_fl[i], MPFR_RNDN);
        if constexpr (O == OP2::ADD) {
            mpfr_add(mr, mx, my, cvt_rm(ctx.rm()));
        } else if constexpr (O == OP2::SUB) {
            mpfr_sub(mr, mx, my, cvt_rm(ctx.rm()));
        } else if constexpr (O == OP2::MUL) {
            mpfr_mul(mr, mx, my, cvt_rm(ctx.rm()));
        } else if constexpr (O == OP2::DIV) {
            mpfr_div(mr, mx, my, cvt_rm(ctx.rm()));
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
    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    std::vector<float> z_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
        z_fl[i] = static_cast<float>(z_vals[i]);
    }

    mpfr_t mx, my, mz, mr;
    mpfr_init2(mx, 24);
    mpfr_init2(my, 24);
    mpfr_init2(mz, 24);
    mpfr_init2(mr, ctx.prec());

    volatile double result = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; i++) {
        mpfr_set_flt(mx, x_fl[i], MPFR_RNDN);
        mpfr_set_flt(my, y_fl[i], MPFR_RNDN);
        mpfr_set_flt(mz, z_fl[i], MPFR_RNDN);
        if constexpr (O == OP3::FMA) {
            mpfr_fma(mr, mx, my, mz, cvt_rm(ctx.rm()));
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

static uint8_t cvt_rm_softfloat(mpfx::RM rm) {
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
double softfloat_op1(const std::vector<double>& x_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    // Quantize to FP32
    std::vector<float> x_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
    }

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        float32_t x;
        x.v = std::bit_cast<uint32_t>(x_fl[i]);
        
        if constexpr (O == OP1::SQRT) {
            float32_t r = f32_sqrt(x);
            result = std::bit_cast<float>(r.v);
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
double softfloat_op2(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        float32_t x, y;
        x.v = std::bit_cast<uint32_t>(x_fl[i]);
        y.v = std::bit_cast<uint32_t>(y_fl[i]);
        
        if constexpr (O == OP2::ADD) {
            float32_t r = f32_add(x, y);
            result = std::bit_cast<float>(r.v);
        } else if constexpr (O == OP2::SUB) {
            float32_t r = f32_sub(x, y);
            result = std::bit_cast<float>(r.v);
        } else if constexpr (O == OP2::MUL) {
            float32_t r = f32_mul(x, y);
            result = std::bit_cast<float>(r.v);
        } else if constexpr (O == OP2::DIV) {
            float32_t r = f32_div(x, y);
            result = std::bit_cast<float>(r.v);
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
double softfloat_op3(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const std::vector<double>& z_vals, const mpfx::Context& ctx, size_t N) {
    softfloat_roundingMode = cvt_rm_softfloat(ctx.rm());

    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    std::vector<float> z_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
        z_fl[i] = static_cast<float>(z_vals[i]);
    }

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        float32_t x, y, z;
        x.v = std::bit_cast<uint32_t>(x_fl[i]);
        y.v = std::bit_cast<uint32_t>(y_fl[i]);
        z.v = std::bit_cast<uint32_t>(z_fl[i]);
        
        if constexpr (O == OP3::FMA) {
            float32_t r = f32_mulAdd(x, y, z);
            result = std::bit_cast<float>(r.v);
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

static Vfpu::RoundingMode cvt_rm_floppyfloat(mpfx::RM rm) {
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
double floppyfloat_op1(const std::vector<double>& x_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    // Quantize to FP32
    std::vector<float> x_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
    }

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP1::SQRT) {
            result = ff.Sqrt(x_fl[i]);
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
double floppyfloat_op2(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP2::ADD) {
            result = ff.Add(x_fl[i], y_fl[i]);
        } else if constexpr (O == OP2::SUB) {
            result = ff.Sub(x_fl[i], y_fl[i]);
        } else if constexpr (O == OP2::MUL) {
            result = ff.Mul(x_fl[i], y_fl[i]);
        } else if constexpr (O == OP2::DIV) {
            result = ff.Div(x_fl[i], y_fl[i]);
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
double floppyfloat_op3(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const std::vector<double>& z_vals, const mpfx::Context& ctx, size_t N) {
    FloppyFloat ff;
    ff.rounding_mode = cvt_rm_floppyfloat(ctx.rm());

    // Quantize to FP32
    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    std::vector<float> z_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
        z_fl[i] = static_cast<float>(z_vals[i]);
    }

    volatile float result = 0.0f;
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        if constexpr (O == OP3::FMA) {
            result = ff.Fma(x_fl[i], y_fl[i], z_fl[i]);
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
void benchmark_op1(
    const mpfx::Context& input_ctx,
    const mpfx::Context& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<double> x_vals(num_inputs);
    if constexpr (O == OP1::SQRT) {
        generate_inputs(x_vals, input_ctx, 0.0, 1.0); // sqrt requires non-negative inputs
    } else {
        generate_inputs(x_vals, input_ctx);
    }

    // Run reference
    const double duration_ref = reference_op1<O>(x_vals, num_inputs);

    // Run MPFR
    const double duration_mpfr = mpfr_op1<O>(x_vals, output_ctx, num_inputs);

    // Run SoftFloat
    const double duration_softfloat = softfloat_op1<O>(x_vals, output_ctx, num_inputs);

    // Run FloppyFloat
    const double duration_floppyfloat = floppyfloat_op1<O>(x_vals, output_ctx, num_inputs);

    // Run MPFX engines
    const double duration_mpfx_rto = mpfx_op1<mpfx::Engine::FP_RTO, O>(x_vals, output_ctx, num_inputs);
    const double duration_mpfx_softfloat = mpfx_op1<mpfx::Engine::SOFTFLOAT, O>(x_vals, output_ctx, num_inputs);
    const double duration_mpfx_ffloat = mpfx_op1<mpfx::Engine::FFLOAT, O>(x_vals, output_ctx, num_inputs);
    const double duration_mpfx_eft = mpfx_op1<mpfx::Engine::EFT, O>(x_vals, output_ctx, num_inputs);

    // Report result
    report_result(
        to_string(O),
        duration_ref, duration_mpfr, duration_softfloat, duration_floppyfloat,
        duration_mpfx_rto, duration_mpfx_softfloat, duration_mpfx_ffloat, duration_mpfx_eft
    );
}

template <OP2 O>
void benchmark_op2(
    const mpfx::Context& input_ctx,
    const mpfx::Context& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<double> x_vals(num_inputs);
    std::vector<double> y_vals(num_inputs);
    generate_inputs(x_vals, input_ctx);
    generate_inputs(y_vals, input_ctx);

    // Run reference
    const double duration_ref = reference_op2<O>(x_vals, y_vals, num_inputs);

    // Run MPFR
    const double duration_mpfr = mpfr_op2<O>(x_vals, y_vals, output_ctx, num_inputs);

    // Run SoftFloat
    const double duration_softfloat = softfloat_op2<O>(x_vals, y_vals, output_ctx, num_inputs);

    // Run FloppyFloat
    const double duration_floppyfloat = floppyfloat_op2<O>(x_vals, y_vals, output_ctx, num_inputs);

    // Run MPFX engines
    const double duration_mpfx_rto = mpfx_op2<mpfx::Engine::FP_RTO, O>(x_vals, y_vals, output_ctx, num_inputs);
    const double duration_mpfx_softfloat = mpfx_op2<mpfx::Engine::SOFTFLOAT, O>(x_vals, y_vals, output_ctx, num_inputs);
    const double duration_mpfx_ffloat = mpfx_op2<mpfx::Engine::FFLOAT, O>(x_vals, y_vals, output_ctx, num_inputs);
    const double duration_mpfx_eft = mpfx_op2<mpfx::Engine::EFT, O>(x_vals, y_vals, output_ctx, num_inputs);

    // Report result
    report_result(
        to_string(O),
        duration_ref, duration_mpfr, duration_softfloat, duration_floppyfloat,
        duration_mpfx_rto, duration_mpfx_softfloat, duration_mpfx_ffloat, duration_mpfx_eft
    );
}

template <OP3 O>
void benchmark_op3(
    const mpfx::Context& input_ctx,
    const mpfx::Context& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<double> x_vals(num_inputs);
    std::vector<double> y_vals(num_inputs);
    std::vector<double> z_vals(num_inputs);
    generate_inputs(x_vals, input_ctx);
    generate_inputs(y_vals, input_ctx);
    generate_inputs(z_vals, input_ctx);

    // Run reference
    const double duration_ref = reference_op3<O>(x_vals, y_vals, z_vals, num_inputs);

    // Run MPFR
    const double duration_mpfr = mpfr_op3<O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);

    // Run SoftFloat
    const double duration_softfloat = softfloat_op3<O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);

    // Run FloppyFloat
    const double duration_floppyfloat = floppyfloat_op3<O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);

    // Run MPFX engines
    const double duration_mpfx_rto = mpfx_op3<mpfx::Engine::FP_RTO, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);
    const double duration_mpfx_softfloat = mpfx_op3<mpfx::Engine::SOFTFLOAT, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);
    const double duration_mpfx_ffloat = mpfx_op3<mpfx::Engine::FFLOAT, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);
    const double duration_mpfx_eft = mpfx_op3<mpfx::Engine::EFT, O>(x_vals, y_vals, z_vals, output_ctx, num_inputs);

    // Report result
    report_result(
        to_string(O),
        duration_ref, duration_mpfr, duration_softfloat, duration_floppyfloat,
        duration_mpfx_rto, duration_mpfx_softfloat, duration_mpfx_ffloat, duration_mpfx_eft
    );
}


int main() {
    const auto INPUT_CTX = mpfx::IEEE754Context(8, 32, mpfx::RM::RNE);
    const auto OUTPUT_CTX = mpfx::IEEE754Context(8, 16, mpfx::RM::RNE);
    constexpr size_t N = 10'000'000;

    report_header();
    benchmark_op2<OP2::ADD>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op2<OP2::SUB>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op2<OP2::MUL>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op2<OP2::DIV>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op1<OP1::SQRT>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op3<OP3::FMA>(INPUT_CTX, OUTPUT_CTX, N);
    return 0;
}
