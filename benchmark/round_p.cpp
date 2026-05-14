#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
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
static double benchmark_round_p(const std::vector<float>& x_vals, mpfx::prec_t p) {
    const size_t n = x_vals.size();
    volatile float result = 0.0f;

    auto start = steady_clock::now();
    for (size_t i = 0; i < n; i++) {
        result = mpfx::experimental::round_p<rm>(x_vals[i], p);
    }
    auto end = steady_clock::now();
    (void) result;

    auto duration = duration_cast<nanoseconds>(end - start).count();
    return static_cast<double>(duration) / static_cast<double>(n);
}

template <mpfx::flag_mask_t FlagMask = mpfx::Flags::NO_FLAGS>
static double benchmark_default(const std::vector<float>& x_vals, mpfx::prec_t p, mpfx::RM rm) {
    const size_t n = x_vals.size();
    volatile float result = 0.0f;

    auto start = steady_clock::now();
    for (size_t i = 0; i < n; i++) {
        result = static_cast<float>(
            mpfx::round<FlagMask>(static_cast<double>(x_vals[i]), p, std::nullopt, rm));
    }
    auto end = steady_clock::now();
    (void) result;

    auto duration = duration_cast<nanoseconds>(end - start).count();
    return static_cast<double>(duration) / static_cast<double>(n);
}

static double dispatch_round_p(const std::vector<float>& x_vals, mpfx::prec_t p, mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE: return benchmark_round_p<mpfx::RM::RNE>(x_vals, p);
        case mpfx::RM::RNA: return benchmark_round_p<mpfx::RM::RNA>(x_vals, p);
        case mpfx::RM::RTP: return benchmark_round_p<mpfx::RM::RTP>(x_vals, p);
        case mpfx::RM::RTN: return benchmark_round_p<mpfx::RM::RTN>(x_vals, p);
        case mpfx::RM::RTZ: return benchmark_round_p<mpfx::RM::RTZ>(x_vals, p);
        case mpfx::RM::RAZ: return benchmark_round_p<mpfx::RM::RAZ>(x_vals, p);
        case mpfx::RM::RTO: return benchmark_round_p<mpfx::RM::RTO>(x_vals, p);
        case mpfx::RM::RTE: return benchmark_round_p<mpfx::RM::RTE>(x_vals, p);
        default:            throw std::runtime_error("invalid rounding mode");
    }
}

int main() {
    using FP32 = mpfx::float_params<float>::params;
    static constexpr size_t N = 20'000'000;

    static constexpr std::array<mpfx::RM, 8> rms = {
        mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP, mpfx::RM::RTN,
        mpfx::RM::RTZ, mpfx::RM::RAZ, mpfx::RM::RTO, mpfx::RM::RTE,
    };
    static constexpr std::array<mpfx::prec_t, 7> ps = { 1, 4, 8, 12, 16, 20, 23 };

    std::cout << "============================================================\n";
    std::cout << "   MPFX round_p vs library round() benchmark\n";
    std::cout << "============================================================\n";
    std::cout << "Operations:   " << N << " per (mode, p) cell\n";
    std::cout << "Input range:  fp32-normal (random sign / 23-bit mantissa /\n";
    std::cout << "              biased exponent in [1, 254])\n";
    std::cout << "------------------------------------------------------------\n\n";

    std::cout << "Generating random test data...\n";
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint32_t> bexp_dist(1, (1u << FP32::E) - 2);
    std::uniform_int_distribution<uint32_t> mant_dist(0, static_cast<uint32_t>(FP32::MMASK));

    std::vector<float> x_vals(N);
    for (size_t i = 0; i < N; i++) {
        const uint32_t s = sign_dist(rng);
        const uint32_t e = bexp_dist(rng);
        const uint32_t m = mant_dist(rng);
        x_vals[i] = std::bit_cast<float>((s << (FP32::M + FP32::E)) | (e << FP32::M) | m);
    }
    std::cout << "Done.\n\n";

    // collect timings so we can also print the speedup table
    std::array<std::array<double, ps.size()>, rms.size()> rp_times{};
    std::array<std::array<double, ps.size()>, rms.size()> lib_times{};
    for (size_t r = 0; r < rms.size(); r++) {
        for (size_t pi = 0; pi < ps.size(); pi++) {
            rp_times[r][pi]  = dispatch_round_p(x_vals, ps[pi], rms[r]);
            lib_times[r][pi] = benchmark_default(x_vals, ps[pi], rms[r]);
        }
    }

    auto print_header = [&]() {
        std::cout << std::left << std::setw(6) << "Mode";
        for (auto p : ps) {
            std::cout << std::right << std::setw(11) << ("p=" + std::to_string(static_cast<int>(p)));
        }
        std::cout << "\n";
        std::cout << "------------------------------------------------------------\n";
    };

    std::cout << "round_p (ns/op):\n";
    std::cout << "------------------------------------------------------------\n";
    print_header();
    for (size_t r = 0; r < rms.size(); r++) {
        std::cout << std::left << std::setw(6) << rm_to_string(rms[r]);
        for (size_t pi = 0; pi < ps.size(); pi++) {
            std::cout << std::right << std::fixed << std::setprecision(2)
                      << std::setw(10) << rp_times[r][pi] << "n";
        }
        std::cout << "\n";
    }

    std::cout << "\nLibrary round() baseline, NO_FLAGS (ns/op):\n";
    std::cout << "------------------------------------------------------------\n";
    print_header();
    for (size_t r = 0; r < rms.size(); r++) {
        std::cout << std::left << std::setw(6) << rm_to_string(rms[r]);
        for (size_t pi = 0; pi < ps.size(); pi++) {
            std::cout << std::right << std::fixed << std::setprecision(2)
                      << std::setw(10) << lib_times[r][pi] << "n";
        }
        std::cout << "\n";
    }

    std::cout << "\nSpeedup (library / round_p):\n";
    std::cout << "------------------------------------------------------------\n";
    print_header();
    for (size_t r = 0; r < rms.size(); r++) {
        std::cout << std::left << std::setw(6) << rm_to_string(rms[r]);
        for (size_t pi = 0; pi < ps.size(); pi++) {
            const double s = lib_times[r][pi] / rp_times[r][pi];
            std::cout << std::right << std::fixed << std::setprecision(2)
                      << std::setw(10) << s << "x";
        }
        std::cout << "\n";
    }
    std::cout << "============================================================\n";

    return 0;
}
