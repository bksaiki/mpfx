#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include <mpfr.h>
#include <mpfx.hpp>

// SoftFloat integration
extern "C" {
    #include "softfloat.h"
}

using namespace std::chrono;

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

static std::string rm_to_string(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE:
            return "RNE (Round to Nearest Even)";
        case mpfx::RM::RTP:
            return "RTP (Round Toward Positive)";
        case mpfx::RM::RTN:
            return "RTN (Round Toward Negative)";
        case mpfx::RM::RTZ:
            return "RTZ (Round to Zero)";
        case mpfx::RM::RAZ:
            return "RAZ (Round Away from Zero)";
        default:
            return "Unknown";
    }
}

// SoftFloat rounding mode conversion
static void set_softfloat_rm(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE:
            softfloat_roundingMode = softfloat_round_near_even;
            break;
        case mpfx::RM::RTP:
            softfloat_roundingMode = softfloat_round_max;
            break;
        case mpfx::RM::RTN:
            softfloat_roundingMode = softfloat_round_min;
            break;
        case mpfx::RM::RTZ:
            softfloat_roundingMode = softfloat_round_minMag;
            break;
        case mpfx::RM::RAZ:
            softfloat_roundingMode = softfloat_round_near_maxMag;
            break;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

double benchmark_mpfx_sqrt(const std::vector<double>& x_vals,
                          int p, mpfx::RM rm) {
    const size_t n = x_vals.size();
    volatile double result = 0.0; // volatile to prevent optimization

    const mpfx::MPContext ctx(p, rm);
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        result = mpfx::sqrt(x_vals[i], ctx);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

double benchmark_mpfr_sqrt(const std::vector<double>& x_vals,
                           int p, mpfx::RM rm) {
    const size_t n = x_vals.size();
    mpfr_t mx, mr;
    
    mpfr_init2(mx, 53);
    mpfr_init2(mr, p);
    
    volatile double result = 0.0; // volatile to prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_sqrt(mr, mx, cvt_rm(rm));
        result = mpfr_get_d(mr, MPFR_RNDN);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    
    mpfr_clear(mx);
    mpfr_clear(mr);
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

// SoftFloat benchmark (32-bit)
double benchmark_softfloat_sqrt(const std::vector<double>& x_vals, mpfx::RM rm) {
    const size_t n = x_vals.size();
    
    set_softfloat_rm(rm);
    volatile double result = 0.0; // volatile to prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        // Convert double to SoftFloat f32, compute sqrt, convert back
        union { float f; uint32_t i; } converter_x, converter_result;
        converter_x.f = static_cast<float>(x_vals[i]);
        float32_t sf_x = { converter_x.i };
        
        float32_t sf_result = f32_sqrt(sf_x);
        
        converter_result.i = sf_result.v;
        result = static_cast<double>(converter_result.f);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

int main() {
    // Configuration
    static constexpr size_t N = 100'000'000; // 10 million operations
    static constexpr int PRECISION = 24;
    static constexpr mpfx::RM ROUNDING_MODE = mpfx::RM::RNE;
    
    std::cout << "=======================================================\n";
    std::cout << "   MPFX vs MPFR vs SoftFloat Square Root Benchmark\n";
    std::cout << "=======================================================\n";
    std::cout << "Operations:     " << N << "\n";
    std::cout << "Precision:      " << PRECISION << " bits (MPFX/MPFR), 32-bit (SoftFloat)\n";
    std::cout << "Rounding mode:  " << rm_to_string(ROUNDING_MODE) << "\n";
    std::cout << "Input range:    [0.1, 10.0] (uniform)\n";
    std::cout << "-------------------------------------------------\n\n";
    
    // Generate random test data (positive values only for sqrt)
    std::cout << "Generating random test data...\n";
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(0.1, 10.0); // positive values only
    
    std::vector<double> x_vals(N);
    
    for (size_t i = 0; i < N; i++) {
        x_vals[i] = dist(rng);
    }
    
    std::cout << "Done.\n\n";
    
    // Benchmark MPFX
    std::cout << "Benchmarking MPFX sqrt()...\n";
    double mpfx_time = benchmark_mpfx_sqrt(x_vals, PRECISION, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Benchmark MPFR
    std::cout << "Benchmarking MPFR mpfr_sqrt()...\n";
    double mpfr_time = benchmark_mpfr_sqrt(x_vals, PRECISION, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Benchmark SoftFloat
    std::cout << "Benchmarking SoftFloat f32_sqrt()...\n";
    double softfloat_time = benchmark_softfloat_sqrt(x_vals, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Results
    std::cout << "=======================================================\n";
    std::cout << "                      RESULTS\n";
    std::cout << "=======================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MPFX sqrt():              " << mpfx_time << " ns/op\n";
    std::cout << "MPFR mpfr_sqrt():        " << mpfr_time << " ns/op\n";
    std::cout << "SoftFloat f32_sqrt():    " << softfloat_time << " ns/op\n";
    std::cout << "-------------------------------------------------------\n";
    
    // Performance comparisons
    if (mpfx_time < mpfr_time) {
        double speedup = mpfr_time / mpfx_time;
        std::cout << "MPFX is " << std::setprecision(2) << speedup << "x FASTER than MPFR\n";
    } else {
        double slowdown = mpfx_time / mpfr_time;
        std::cout << "MPFX is " << std::setprecision(2) << slowdown << "x SLOWER than MPFR\n";
    }
    
    if (mpfx_time < softfloat_time) {
        double speedup = softfloat_time / mpfx_time;
        std::cout << "MPFX is " << std::setprecision(2) << speedup << "x FASTER than SoftFloat\n";
    } else {
        double slowdown = mpfx_time / softfloat_time;
        std::cout << "MPFX is " << std::setprecision(2) << slowdown << "x SLOWER than SoftFloat\n";
    }
    
    if (mpfr_time < softfloat_time) {
        double speedup = softfloat_time / mpfr_time;
        std::cout << "MPFR is " << std::setprecision(2) << speedup << "x FASTER than SoftFloat\n";
    } else {
        double slowdown = mpfr_time / softfloat_time;
        std::cout << "MPFR is " << std::setprecision(2) << slowdown << "x SLOWER than SoftFloat\n";
    }
    std::cout << "=======================================================\n";
    
    return 0;
}
