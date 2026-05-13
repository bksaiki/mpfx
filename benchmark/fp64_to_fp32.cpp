#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <mpfx.hpp>

using namespace std::chrono;

static std::string rm_to_string(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE: return "RNE";
        case mpfx::RM::RNA: return "RNA";
        case mpfx::RM::RTP: return "RTP";
        case mpfx::RM::RTN: return "RTN";
        case mpfx::RM::RTZ: return "RTZ";
        case mpfx::RM::RAZ: return "RAZ";
        case mpfx::RM::RTO: return "RTO";
        case mpfx::RM::RTE: return "RTE";
        default:            return "???";
    }
}

template <mpfx::RM rm>
static double benchmark_optimized(const std::vector<double>& x_vals) {
    const size_t n = x_vals.size();
    volatile float result = 0.0f;

    auto start = steady_clock::now();
    for (size_t i = 0; i < n; i++) {
        result = mpfx::experimental::fp64_to_fp32<rm>(x_vals[i]);
    }
    auto end = steady_clock::now();
    (void) result;

    auto duration = duration_cast<nanoseconds>(end - start).count();
    return static_cast<double>(duration) / static_cast<double>(n);
}

template <mpfx::flag_mask_t FlagMask = mpfx::Flags::NO_FLAGS>
static double benchmark_default(const std::vector<double>& x_vals, mpfx::RM rm) {
    const size_t n = x_vals.size();
    const mpfx::IEEE754Context ctx(8, 32, rm);
    volatile float result = 0.0f;

    auto start = steady_clock::now();
    for (size_t i = 0; i < n; i++) {
        result = static_cast<float>(mpfx::round<FlagMask>(x_vals[i], ctx));
    }
    auto end = steady_clock::now();
    (void) result;

    auto duration = duration_cast<nanoseconds>(end - start).count();
    return static_cast<double>(duration) / static_cast<double>(n);
}

static double dispatch_optimized(const std::vector<double>& x_vals, mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE: return benchmark_optimized<mpfx::RM::RNE>(x_vals);
        case mpfx::RM::RNA: return benchmark_optimized<mpfx::RM::RNA>(x_vals);
        case mpfx::RM::RTP: return benchmark_optimized<mpfx::RM::RTP>(x_vals);
        case mpfx::RM::RTN: return benchmark_optimized<mpfx::RM::RTN>(x_vals);
        case mpfx::RM::RTZ: return benchmark_optimized<mpfx::RM::RTZ>(x_vals);
        case mpfx::RM::RAZ: return benchmark_optimized<mpfx::RM::RAZ>(x_vals);
        case mpfx::RM::RTO: return benchmark_optimized<mpfx::RM::RTO>(x_vals);
        case mpfx::RM::RTE: return benchmark_optimized<mpfx::RM::RTE>(x_vals);
        default:            throw std::runtime_error("invalid rounding mode");
    }
}

int main() {
    static constexpr size_t N = 100'000'000;

    // exercise the full fp32 range, including overflow and subnormal regions
    static constexpr std::array<mpfx::RM, 8> rms = {
        mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP, mpfx::RM::RTN,
        mpfx::RM::RTZ, mpfx::RM::RAZ, mpfx::RM::RTO, mpfx::RM::RTE,
    };

    std::cout << "============================================================\n";
    std::cout << "   MPFX fp64_to_fp32 vs default round() benchmark\n";
    std::cout << "============================================================\n";
    std::cout << "Operations:   " << N << " per rounding mode\n";
    std::cout << "Format:       binary32 (es=8, nbits=32)\n";
    std::cout << "Input range:  random sign / 23-bit mantissa / exp in [-160, 110]\n";
    std::cout << "------------------------------------------------------------\n\n";

    std::cout << "Generating random test data...\n";
    std::random_device rd;
    std::mt19937_64 rng(rd());

    std::uniform_int_distribution<int> sign_dist(0, 1);
    std::uniform_int_distribution<uint32_t> c_dist(0, 0x7fffff);
    std::uniform_int_distribution<mpfx::exp_t> exp_dist(-160, 110);

    std::vector<double> x_vals(N);
    for (size_t i = 0; i < N; i++) {
        // generate a random float
        const bool s = sign_dist(rng) == 1;
        const mpfx::exp_t exp = exp_dist(rng);
        const uint32_t c = c_dist(rng);
        x_vals[i] = mpfx::make_float<double>(s, exp, c);
    }
    std::cout << "Done.\n\n";

    std::cout << std::left << std::setw(6) << "Mode"
              << std::right << std::setw(16) << "Optimized (ns)"
              << std::setw(16) << "Default (ns)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << "------------------------------------------------------------\n";

    for (const auto& rm : rms) {
        const double opt_time = dispatch_optimized(x_vals, rm);
        const double def_time = benchmark_default(x_vals, rm);
        const double speedup = def_time / opt_time;

        std::cout << std::left << std::setw(6) << rm_to_string(rm)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(16) << opt_time
                  << std::setw(16) << def_time
                  << std::setw(11) << speedup << "x"
                  << "\n";
    }
    std::cout << "============================================================\n";

    return 0;
}
