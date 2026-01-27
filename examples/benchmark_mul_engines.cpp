#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <mpfx.hpp>

#ifdef USE_SOFTFLOAT
extern "C" {
#include <softfloat.h>
}
#endif

using namespace std::chrono;

// Global configuration
static constexpr size_t N = 100'000'000; // 100 million operations

// Global rounding context (target precision for multiplication)
static const mpfx::IEEE754Context ROUND_CTX(8, 24, mpfx::RM::RNE); // FP32

// Global input context for sampling
static const mpfx::IEEE754Context INPUT_CTX(8, 24, mpfx::RM::RNE); // FP32

static void generate_inputs(std::vector<double>& x_vals, std::vector<double>& y_vals) {
    std::cout << "Generating " << N << " random test pairs...\n";
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (size_t i = 0; i < N; i++) {
        // Sample and quantize inputs according to input context
        x_vals[i] = mpfx::round(dist(rng), INPUT_CTX);
        y_vals[i] = mpfx::round(dist(rng), INPUT_CTX);
    }
    std::cout << "Input generation complete.\n\n";
}

static double run_reference(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running reference (native double multiplication)...\n";

    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    auto start = steady_clock::now();

    volatile float result;
    for (size_t i = 0; i < N; i++) {
        result = x_fl[i] * y_fl[i];
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_softfloat(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running SoftFloat reference...\n";

    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    auto start = steady_clock::now();

    volatile float result;
    for (size_t i = 0; i < N; i++) {
        // Convert to SoftFloat format
        float32_t x, y;
        x.v = std::bit_cast<uint32_t>(x_fl[i]);
        y.v = std::bit_cast<uint32_t>(y_fl[i]);

        // Perform multiplication
        float32_t r = f32_mul(x, y);

        // Store result
        result = std::bit_cast<float>(r.v);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_floppyfloat(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running FloppyFloat reference...\n";

    FloppyFloat ff;
    ff.rounding_mode = Vfpu::kRoundTiesToEven;

    std::vector<float> x_fl(N);
    std::vector<float> y_fl(N);
    for (size_t i = 0; i < N; i++) {
        x_fl[i] = static_cast<float>(x_vals[i]);
        y_fl[i] = static_cast<float>(y_vals[i]);
    }

    auto start = steady_clock::now();

    volatile float result;
    for (size_t i = 0; i < N; i++) {
        result = ff.Mul(x_fl[i], y_fl[i]);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_rto_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running RTO engine...\n";

    auto start = steady_clock::now();

    volatile double result;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::mul<mpfx::EngineType::FP_RTO>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_exact_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running EXACT engine...\n";

    auto start = steady_clock::now();

    volatile double result;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::mul<mpfx::EngineType::FP_EXACT>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_fixed_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running FIXED engine...\n";

    auto start = steady_clock::now();

    volatile double result;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::mul<mpfx::EngineType::FIXED>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_softfloat_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running SoftFloat engine...\n";

    auto start = steady_clock::now();

    volatile double result;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::mul<mpfx::EngineType::SOFTFLOAT>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_floppyfloat_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running FloppyFloat engine...\n";

    auto start = steady_clock::now();

    volatile double result;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::mul<mpfx::EngineType::FFLOAT>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_eft_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running EFT engine...\n";

    auto start = steady_clock::now();

    volatile double result;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::mul<mpfx::EngineType::EFT>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = steady_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}


int main() {
    std::cout << "=== Multiplication Engine Benchmark ===\n";
    std::cout << "Operations: " << N << "\n";
    std::cout << "Rounding context: FP32\n";
    std::cout << "Input context: FP32\n\n";

    // Generate inputs
    std::vector<double> x_vals(N);
    std::vector<double> y_vals(N);
    generate_inputs(x_vals, y_vals);

    // Run references
    const double duration_ref = run_reference(x_vals, y_vals);
    const double duration_softfloat_ref = run_softfloat(x_vals, y_vals);
    const double duration_floppyfloat_ref = run_floppyfloat(x_vals, y_vals);

    // Run engines
    const double duration_rto = run_rto_engine(x_vals, y_vals);
    const double duration_exact = run_exact_engine(x_vals, y_vals);
    const double duration_fixed = run_fixed_engine(x_vals, y_vals);
    const double duration_softfloat = run_softfloat_engine(x_vals, y_vals);
    const double duration_floppyfloat = run_floppyfloat_engine(x_vals, y_vals);
    const double duration_eft = run_eft_engine(x_vals, y_vals);

    // Print summary
    std::cout << "=== Performance Summary ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Reference:     " << duration_ref * 1e-6 << "s (baseline)\n";
    std::cout << "SoftFloat:     " << duration_softfloat_ref * 1e-6 << "s (" 
              << duration_softfloat_ref / duration_ref << "x slowdown)\n";
    std::cout << "FloppyFloat:   " << duration_floppyfloat_ref * 1e-6 << "s (" 
              << duration_floppyfloat_ref / duration_ref << "x slowdown)\n";
    std::cout << "RTO engine:    " << duration_rto * 1e-6 << "s (" 
              << duration_rto / duration_ref << "x slowdown)\n";
    std::cout << "EXACT engine:  " << duration_exact * 1e-6 << "s (" 
              << duration_exact / duration_ref << "x slowdown)\n";
    std::cout << "FIXED engine:  " << duration_fixed * 1e-6 << "s (" 
              << duration_fixed / duration_ref << "x slowdown)\n";
    std::cout << "SoftFloat:     " << duration_softfloat * 1e-6 << "s (" 
              << duration_softfloat / duration_ref << "x slowdown)\n";
    std::cout << "FloppyFloat:   " << duration_floppyfloat * 1e-6 << "s (" 
              << duration_floppyfloat / duration_ref << "x slowdown)\n";
    std::cout << "EFT engine:    " << duration_eft * 1e-6 << "s (" 
              << duration_eft / duration_ref << "x slowdown)\n";

    return 0;
}
