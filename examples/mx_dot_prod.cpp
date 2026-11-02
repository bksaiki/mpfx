/**
 * @file mx_dot_prod.cpp
 * @brief MX dot product: SoftFloat vs. MPFX.
 *
 * Draws pairs of standard-normal vectors of length 4096 and quantizes each
 * into OCP MX blocks of 32 elements with a shared power-of-two scale,
 * rounding to nearest, ties to even, and saturating at the element format's
 * largest value. For each pairing of element formats (E5M2 x E5M2,
 * E5M2 x E4M3, E4M3 x E4M3), it computes the dot product of every pair two
 * ways, with SoftFloat (`mx_dot_prod_sf`) and with MPFX (`mx_dot_prod_impl`):
 * each block's exact dot product, scaled, is added into an FP32 accumulator
 * rounding to nearest, ties to even. Only those two loops are timed.
 *
 * The two implementations must then agree bit for bit, with a finite result,
 * on every input; a pairing that does not prints FAIL in place of its times,
 * and the program exits 1. On 23,000 drawn inputs, a version that adds each
 * block's dot product to the accumulator in plain double, then rounds to
 * FP32, passes the check too.
 *
 * Build with CMake and -DBUILD_EXAMPLES=ON (needs SoftFloat; see
 * install_softfloat.sh), then run
 *
 *     mx_dot_prod <num_inputs> [seed]
 *
 * The seed defaults to 1. Memory grows by 130 to 160 KB per input (vectors
 * and blocks), so 100,000 inputs need 13 to 16 GB.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

#include <mpfx.hpp>

extern "C" {
#include <softfloat.h>
}

enum class MX {
    E5M2,
    E4M3,
    E2M3,
    E3M2,
    E2M1
};

template <MX T>
struct mx_params {};

template <>
struct mx_params<MX::E5M2> {
    static constexpr int P = 3;
    static constexpr int NMIN = -17;
    static constexpr double MAX = 57344.0;
};

template <>
struct mx_params<MX::E4M3> {
    static constexpr int P = 4;
    static constexpr int NMIN = -10;
    static constexpr double MAX = 448.0;
    
};

template <>
struct mx_params<MX::E3M2> {
    static constexpr int P = 3;
    static constexpr int NMIN = -5;
    static constexpr double MAX = 28.0;
};

template <>
struct mx_params<MX::E2M3> {
    static constexpr int P = 4;
    static constexpr int NMIN = -4;
    static constexpr double MAX = 7.5;
};

template <>
struct mx_params<MX::E2M1> {
    static constexpr int P = 2;
    static constexpr int NMIN = -2;
    static constexpr double MAX = 6.0;
};

struct TimingResult {
    std::string config;
    double sf_time;
    double mpfx_time;
    size_t num_mismatches;
};

constexpr const char* mx_name(MX m) {
    switch (m) {
        case MX::E5M2: return "E5M2";
        case MX::E4M3: return "E4M3";
        case MX::E3M2: return "E3M2";
        case MX::E2M3: return "E2M3";
        case MX::E2M1: return "E2M1";
    }
    return "???";
}

// Exact power of two, 2^e, as a double. Constexpr so that fixed-point scale
// factors fold into a single multiply instead of an out-of-line ldexp call.
constexpr double exp2i(int e) {
    double r = 1.0;
    const double base = e < 0 ? 0.5 : 2.0;
    for (int i = 0, n = e < 0 ? -e : e; i < n; i++) {
        r *= base;
    }
    return r;
}


using mx_block_t = std::tuple<double, std::array<double, 32>>;

template <size_t K, MX F, size_t N>
std::vector<mx_block_t> mx_block_quantize(const std::array<double, N>& vec) {
    // number of blocks (last block may be smaller)
    constexpr size_t num_blocks = (N + K - 1) / K;

    // rounding context
    constexpr int P = mx_params<F>::P;
    constexpr int NMIN = mx_params<F>::NMIN;
    constexpr int MAX = mx_params<F>::MAX;
    int EMAX = std::ilogb(MAX);

    // Round to nearest and saturate, so an element past MAX becomes MAX
    // rather than infinity, which would make the dot product meaningless.
    const mpfx::Context ctx(P, NMIN, MAX, mpfx::RM::RNE, mpfx::OverflowMode::TO_MAXVAL);

    // quantize vector into blocks
    std::vector<mx_block_t> blocks(num_blocks);
    for (size_t i = 0; i < num_blocks; i++) {
        const size_t start_idx = i * K;
        const size_t end_idx = std::min(start_idx + K, N);
            
        // compute largest exponent
        int max_e = 0;
        bool e_valid = false;
        for (size_t j = start_idx; j < end_idx; j++) {
            if (std::isfinite(vec[j]) && vec[j] != 0.0) {
                const int e = std::ilogb(std::abs(vec[j]));
                if (e_valid) {
                    max_e = std::max(max_e, e);
                } else {
                    max_e = e;
                    e_valid = true;
                }
            }
        }

        // scale factor is 2 ** max_e
        const double scale = std::ldexp(1.0, max_e - EMAX);

        // divide by scale and quantize
        std::array<double, K> block{};
        for (size_t j = start_idx; j < end_idx; j++) {
            block[j - start_idx] = mpfx::div<mpfx::Engine::EFT>(vec[j], scale, ctx);
        }

        blocks[i] = std::make_tuple(scale, block);
    }

    return blocks;
}

template <MX FA, MX FB>
double mx_dot_prod_sf(const std::vector<mx_block_t>& a_blocks, const std::vector<mx_block_t>& b_blocks) {
    MPFX_ASSERT(a_blocks.size() == b_blocks.size(), "block size mismatch");

    float32_t result {};
    for (size_t i = 0; i < a_blocks.size(); i++) {
        // unpack blocks
        const auto [a_scale, a_elts] = a_blocks[i];
        const auto [b_scale, b_elts] = b_blocks[i];

        // multiply scales
        const double scale = a_scale * b_scale;

        // compute inner dot product
        float128_t scaled_prod;
        if constexpr (FA == MX::E5M2 || FB == MX::E5M2) {
            // need to use FP128 uniformly

            // compute unscaled dot product using FP128
            float128_t prod {};
            for (size_t j = 0; j < a_elts.size(); j++) {
                const double p = a_elts[j] * b_elts[j];

                float128_t p128;
                f64_to_f128M(std::bit_cast<float64_t>(p), &p128);
                f128M_add(&prod, &p128, &prod);
            }

            // apply scale
            float128_t scale128;
            f64_to_f128M(std::bit_cast<float64_t>(scale), &scale128);
            f128M_mul(&prod, &scale128, &scaled_prod);
        } else {
            // can use double-precision uniformly
            double prod = 0.0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                prod += a_elts[j] * b_elts[j];
            }

            // apply scale
            prod *= scale;

            // convert to FP128 for final accumulation
            f64_to_f128M(std::bit_cast<float64_t>(prod), &scaled_prod);
        }

        float128_t result128, sum128;
        f32_to_f128M(result, &result128);

        // perform the addition by double rounding
        softfloat_roundingMode = softfloat_round_odd;
        f128M_add(&result128, &scaled_prod, &sum128);

        // round to nearest with ties to even to get final result
        softfloat_roundingMode = softfloat_round_near_even;
        result = f128M_to_f32(&sum128);
    }

    return static_cast<double>(std::bit_cast<float>(result));
}

template <MX FA, MX FB>
double mx_dot_prod_impl(const std::vector<mx_block_t>& a_blocks, const std::vector<mx_block_t>& b_blocks) {
    constexpr auto A_EXPMIN = mx_params<FA>::NMIN + 1;
    constexpr auto B_EXPMIN = mx_params<FB>::NMIN + 1;
    MPFX_ASSERT(a_blocks.size() == b_blocks.size(), "block size mismatch");

    // rounding contexts
    const mpfx::IEEE754Context accum_ctx(8, 32, mpfx::RM::RNE);

    double result = 0.0;
    for (size_t i = 0; i < a_blocks.size(); i++) {
        // unpack blocks
        const auto [a_scale, a_elts] = a_blocks[i];
        const auto [b_scale, b_elts] = b_blocks[i];

        // multiply scales
        const double scale = a_scale * b_scale;

        // compute scaled dot product
        if constexpr (FA == MX::E5M2 && FB == MX::E5M2) {
            // unscaled dot product should be performed exactly. Each element is an
            // exact integer multiple of 2^EXPMIN, so convert to int64 (never larger
            // than ~2^32) and multiply into int128. The product can reach
            // ~57344^2 * 2^32 > INT64_MAX, so it must be accumulated in 128 bits.
            constexpr auto A_TO_FIXED = exp2i(-A_EXPMIN);
            constexpr auto B_TO_FIXED = exp2i(-B_EXPMIN);

            mpfx::int128_t prod = 0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                const int64_t a = static_cast<int64_t>(a_elts[j] * A_TO_FIXED);
                const int64_t b = static_cast<int64_t>(b_elts[j] * B_TO_FIXED);
                prod += static_cast<mpfx::int128_t>(a) * b;
            }

            // break up prod into 2 parts of 40 and 29 digits
            static constexpr int SPLIT = 40;
            static constexpr mpfx::uint128_t MASK = (mpfx::uint128_t(1) << SPLIT) - 1;

            // decompose
            const double sgn = prod < 0 ? -1.0 : 1.0;
            const mpfx::uint128_t abs_prod = prod < 0 ? static_cast<mpfx::uint128_t>(-prod) : static_cast<mpfx::uint128_t>(prod);
            const mpfx::uint128_t prod_hi = abs_prod >> SPLIT;
            const mpfx::uint128_t prod_lo = abs_prod & MASK;

            // convert to double with scaling (apply sign to all parts uniformly)
            double scaled_hi = sgn * static_cast<double>(prod_hi) * std::ldexp(1.0, A_EXPMIN + B_EXPMIN + SPLIT);
            double scaled_lo = sgn * static_cast<double>(prod_lo) * std::ldexp(1.0, A_EXPMIN + B_EXPMIN);

            // scale the product
            scaled_hi *= scale;
            scaled_lo *= scale;

            // perform `scale * prod + result` using error-free transformations
            result = mpfx::add3(scaled_hi, scaled_lo, result, accum_ctx);
        } else if constexpr ((FA == MX::E5M2 && FB == MX::E4M3) || (FA == MX::E4M3 && FB == MX::E5M2)) {
            // unscaled dot product should be performed with at least 2*P bits of
            // precision. The product fits in int64, and multiplying by the constant
            // 2^-(A_EXPMIN+B_EXPMIN) folds at compile time (vs. an out-of-line ldexp).
            constexpr double TO_FIXED = exp2i(-(A_EXPMIN + B_EXPMIN));

            int64_t prod = 0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                const double p = a_elts[j] * b_elts[j];
                prod += static_cast<int64_t>(p * TO_FIXED);
            }

            // break up prod into 2 parts of 40 and 15 digits
            static constexpr int SPLIT = 40;
            static constexpr mpfx::uint128_t MASK = (mpfx::uint128_t(1) << SPLIT) - 1;

            // break up prod into 2 parts so that we can convert to double without rounding
            // decompose absolute value to avoid cancellation in EFT
            const double sgn = prod < 0 ? -1.0 : 1.0;
            const uint64_t abs_prod = prod < 0 ? static_cast<uint64_t>(-prod) : static_cast<uint64_t>(prod);
            const uint64_t prod_hi = abs_prod >> SPLIT;
            const uint64_t prod_lo = abs_prod & MASK;

            // convert to double with scaling (apply sign to both parts uniformly)
            double scaled_hi = sgn * static_cast<double>(prod_hi) * std::ldexp(1.0, A_EXPMIN + B_EXPMIN + SPLIT);
            double scaled_lo = sgn * static_cast<double>(prod_lo) * std::ldexp(1.0, A_EXPMIN + B_EXPMIN);

            // scale the product
            scaled_hi *= scale;
            scaled_lo *= scale;

            // perform `scale * prod + result` using error-free transformations
            result = mpfx::add3(scaled_hi, scaled_lo, result, accum_ctx);
        } else {
            // compute unscaled dot product in double precision, exact for these formats
            double prod = 0.0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                prod += a_elts[j] * b_elts[j];
            }

            // scale the dot product and add to the result
            result = mpfx::fma<mpfx::Engine::EFT>(scale, prod, result, accum_ctx);
        }
    }

    return result;
}

template <MX FA, MX FB, size_t LEN>
TimingResult time_dot_prod(
    const std::vector<std::array<double, LEN>>& x_vals,
    const std::vector<std::array<double, LEN>>& y_vals
) {
    const size_t N = x_vals.size();
    std::string config = std::string(mx_name(FA)) + "x" + mx_name(FB);
    std::cout << "Timing dot product for config " << config << " with " << N << " inputs..." << std::endl;

    // Quantize data into blocks
    std::vector<std::vector<mx_block_t>> x_blocks(N);
    std::vector<std::vector<mx_block_t>> y_blocks(N);
    for (size_t i = 0; i < N; i++) {
        x_blocks[i] = mx_block_quantize<32, FA>(x_vals[i]);
        y_blocks[i] = mx_block_quantize<32, FB>(y_vals[i]);
    } 

    // Time softfloat reference implementation
    double sf_time;
    std::vector<double> sf_results(N);
    {
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < N; i++) {
            sf_results[i] = mx_dot_prod_sf<FA, FB>(x_blocks[i], y_blocks[i]);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> sf_duration = end - start;
        sf_time = sf_duration.count();
    }

    // Time MPFX implementation
    double mpfx_time;
    std::vector<double> mpfx_results(N);
    {
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < N; i++) {
            mpfx_results[i] = mx_dot_prod_impl<FA, FB>(x_blocks[i], y_blocks[i]);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;
        mpfx_time = duration.count();
    }

    // Check the timed results: the two implementations compute the same
    // function, so they must agree bit for bit, and on these inputs a
    // correct result is always finite. A fast wrong result is no speedup.
    size_t num_mismatches = 0;
    for (size_t i = 0; i < N; i++) {
        const bool agree = std::isfinite(sf_results[i])
            && std::bit_cast<uint64_t>(sf_results[i]) == std::bit_cast<uint64_t>(mpfx_results[i]);
        if (!agree) {
            if (num_mismatches == 0) {
                std::ostringstream first;
                first << std::setprecision(17) << "  mismatch at input " << i
                      << ": sf=" << sf_results[i] << " mpfx=" << mpfx_results[i];
                std::cout << first.str() << std::endl;
            }
            num_mismatches++;
        }
    }

    if (num_mismatches > 0) {
        std::cout << "  " << num_mismatches << " of " << N
                  << " results differ or are not finite" << std::endl;
    }

    return { config, sf_time, mpfx_time, num_mismatches };
}


int main(int argc, char* argv[]) {
    // Command line arguments: number of inputs to test, and an optional seed
    // for them, fixed by default so a run can be repeated. The same seed gives
    // the same inputs only under the same standard library, since the
    // algorithm behind std::normal_distribution varies between them.
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <num_inputs> [seed]" << std::endl;
        return 1;
    }

    const size_t N = std::stoul(argv[1]);
    const uint64_t seed = argc == 3 ? std::stoull(argv[2]) : 1;
    std::cout << "Seed: " << seed << std::endl;

    // Configuration
    static constexpr size_t LEN = 4096;

    // Generate random test data
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Random data
    std::vector<std::array<double, LEN>> x_vals(N);
    std::vector<std::array<double, LEN>> y_vals(N);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < LEN; j++) {
            x_vals[i][j] = dist(rng);
            y_vals[i][j] = dist(rng);
        }
    }

    // Time implementations
    std::vector<TimingResult> results;
    results.push_back(time_dot_prod<MX::E5M2, MX::E5M2>(x_vals, y_vals));
    results.push_back(time_dot_prod<MX::E5M2, MX::E4M3>(x_vals, y_vals));
    results.push_back(time_dot_prod<MX::E4M3, MX::E4M3>(x_vals, y_vals));
    // results.push_back(time_dot_prod<MX::E3M2, MX::E3M2>(x_vals, y_vals));
    // results.push_back(time_dot_prod<MX::E2M3, MX::E2M3>(x_vals, y_vals));
    // results.push_back(time_dot_prod<MX::E2M1, MX::E2M1>(x_vals, y_vals));

    // Print table; a configuration whose results fail the check shows no times
    const int cw = 14; // column width
    std::cout << std::left << std::setw(cw) << "Config"
              << "| " << std::setw(cw) << "SoftFloat"
              << "| " << std::setw(cw) << "MPFX"
              << "| " << std::setw(cw) << "Speedup" << "|" << std::endl;
    std::cout << std::string(cw, '-')
              << "|" << std::string(cw + 1, '-')
              << "|" << std::string(cw + 1, '-')
              << "|" << std::string(cw + 1, '-') << "|" << std::endl;
    bool all_pass = true;
    for (const auto& r : results) {
        if (r.num_mismatches > 0) {
            all_pass = false;
            std::cout << std::left << std::setw(cw) << r.config
                      << "| " << std::setw(3 * cw + 4) << "FAIL: results differ or are not finite"
                      << "|" << std::endl;
            continue;
        }

        const double speedup = r.sf_time / r.mpfx_time;
        std::ostringstream sf_str, mpfx_str, speedup_str;
        sf_str << std::fixed << std::setprecision(4) << r.sf_time << "s";
        mpfx_str << std::fixed << std::setprecision(4) << r.mpfx_time << "s";
        speedup_str << std::fixed << std::setprecision(2) << speedup << "x";
        std::cout << std::left << std::setw(cw) << r.config
                  << "| " << std::setw(cw) << sf_str.str()
                  << "| " << std::setw(cw) << mpfx_str.str()
                  << "| " << std::setw(cw) << speedup_str.str() << "|" << std::endl;
    }

    return all_pass ? 0 : 1;
}
