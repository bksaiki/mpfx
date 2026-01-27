/**
 * @file ops.cpp
 * @brief Benchmarking against other number libraries.
 */

#include <chrono>
#include <iostream>
#include <concepts>
#include <random>
#include <vector>

#include <mpfr.h>
#include <mpfx.hpp>
#include <softfloat.h>
#include <floppy_float.h>

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
static void generate_inputs(std::vector<T>& vals, const mpfx::MPBContext& ctx) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < vals.size(); i++) {
        vals[i] = static_cast<T>(mpfx::round(dist(rng), ctx));
    }
}

static void report_result(
    const std::string& op_name,
    double duration_ref,
    double duration_mpfr
) {
    std::cout << op_name
        << ", " << duration_ref
        << ", " << duration_mpfr
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
double mpfr_op1(const std::vector<double>& x_vals, const mpfx::MPBContext& ctx, size_t N) {
    mpfr_t mx, mr;
    
    mpfr_init2(mx, 53);
    mpfr_init2(mr, ctx.prec());
    
    volatile double result = 0.0;
    
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
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
double mpfr_op2(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const mpfx::MPBContext& ctx, size_t N) {
    mpfr_t mx, my, mr;
    
    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, ctx.prec());
    
    volatile double result = 0.0;
    
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_d(my, y_vals[i], MPFR_RNDN);
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
double mpfr_op3(const std::vector<double>& x_vals, const std::vector<double>& y_vals, const std::vector<double>& z_vals, const mpfx::MPBContext& ctx, size_t N) {
    mpfr_t mx, my, mz, mr;
    
    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mz, 53);
    mpfr_init2(mr, ctx.prec());
    
    volatile double result = 0.0;
    
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < N; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_d(my, y_vals[i], MPFR_RNDN);
        mpfr_set_d(mz, z_vals[i], MPFR_RNDN);
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
// Benchmarking functions

template <OP1 O>
void benchmark_op1(
    const mpfx::MPBContext& input_ctx,
    const mpfx::MPBContext& output_ctx,
    size_t num_inputs
) {
    // Generate inputs
    std::vector<double> x_vals(num_inputs);
    generate_inputs(x_vals, input_ctx);

    // Run reference
    const double duration_ref = reference_op1<O>(x_vals, num_inputs);

    // Run MPFR
    const double duration_mpfr = mpfr_op1<O>(x_vals, output_ctx, num_inputs);

    // Report result
    report_result(to_string(O), duration_ref, duration_mpfr);
}

template <OP2 O>
void benchmark_op2(
    const mpfx::MPBContext& input_ctx,
    const mpfx::MPBContext& output_ctx,
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

    // Report result
    report_result(to_string(O), duration_ref, duration_mpfr);
}

template <OP3 O>
void benchmark_op3(
    const mpfx::MPBContext& input_ctx,
    const mpfx::MPBContext& output_ctx,
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

    // Report result
    report_result(to_string(O), duration_ref, duration_mpfr);
}


int main() {
    const auto INPUT_CTX = mpfx::IEEE754Context(8, 32, mpfx::RM::RNE);
    const auto OUTPUT_CTX = mpfx::IEEE754Context(8, 32, mpfx::RM::RNE);
    static constexpr size_t N = 1'000'000;

    benchmark_op2<OP2::ADD>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op2<OP2::SUB>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op2<OP2::MUL>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op2<OP2::DIV>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op1<OP1::SQRT>(INPUT_CTX, OUTPUT_CTX, N);
    benchmark_op3<OP3::FMA>(INPUT_CTX, OUTPUT_CTX, N);

    return 0;
}
