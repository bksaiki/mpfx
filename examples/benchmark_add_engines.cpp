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
static const mpfx::IEEE754Context ROUND_CTX(8, 32, mpfx::RM::RNE); // FP32

// Global input context for sampling
static const mpfx::IEEE754Context INPUT_CTX(8, 32, mpfx::RM::RNE); // FP32

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
    std::cout << "Running reference (native double addition)...\n";
    auto start = high_resolution_clock::now();

    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += x_vals[i] * y_vals[i];
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "  Result checksum: " << sum << "\n";
    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_rto_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running RTO engine...\n";
    auto start = high_resolution_clock::now();

    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += mpfx::add<mpfx::EngineType::FP_RTO>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "  Result checksum: " << sum << "\n";
    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_softfloat_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running SoftFloat engine...\n";
    auto start = high_resolution_clock::now();

    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += mpfx::add<mpfx::EngineType::SOFTFLOAT>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "  Result checksum: " << sum << "\n";
    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

static double run_floppyfloat_engine(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Running FloppyFloat engine...\n";
    auto start = high_resolution_clock::now();

    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += mpfx::add<mpfx::EngineType::FFLOAT>(x_vals[i], y_vals[i], ROUND_CTX);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "  Result checksum: " << sum << "\n";
    std::cout << "  Duration: " << duration * 1e-6 << " seconds\n\n";
    return duration;
}

int main() {
    std::cout << "=== Addition Engine Benchmark ===\n";
    std::cout << "Operations: " << N << "\n";
    std::cout << "Rounding context: FP32\n";
    std::cout << "Input context: FP32\n\n";

    // Generate inputs
    std::vector<double> x_vals(N);
    std::vector<double> y_vals(N);
    generate_inputs(x_vals, y_vals);

    // Run benchmarks
    const double duration_ref = run_reference(x_vals, y_vals);
    const double duration_rto = run_rto_engine(x_vals, y_vals);
    const double duration_softfloat = run_softfloat_engine(x_vals, y_vals);
#ifdef USE_FLOPPYFLOAT
    const double duration_floppyfloat = run_floppyfloat_engine(x_vals, y_vals);
#endif

    // Print summary
    std::cout << "=== Performance Summary ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Reference:     " << duration_ref * 1e-6 << "s (baseline)\n";
    std::cout << "RTO engine:    " << duration_rto * 1e-6 << "s (" 
              << duration_rto / duration_ref << "x slowdown)\n";
    std::cout << "SoftFloat:     " << duration_softfloat * 1e-6 << "s (" 
              << duration_softfloat / duration_ref << "x slowdown)\n";
#ifdef USE_FLOPPYFLOAT
    std::cout << "FloppyFloat:   " << duration_floppyfloat * 1e-6 << "s (" 
              << duration_floppyfloat / duration_ref << "x slowdown)\n";
#endif

    return 0;
}
