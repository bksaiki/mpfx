#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
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


template <std::floating_point T>
inline std::tuple<T, T> fast_two_sum(T x, T y) {
    const T s = x + y;
    const T yy = s - x;
    const T t = y - yy;
    return { s, t };
}

template <std::floating_point T>
inline std::tuple<T, T> two_sum(T x, T y) {
    const bool swap = std::fabs(x) < std::fabs(y);
    const T a = swap ? y : x;
    const T b = swap ? x : y;
    return fast_two_sum(a, b);
}

// Sorted: if true, assumes |a[0]| >= |a[1]| (uses fast_two_sum for first pair)
template <bool Sorted = false, std::floating_point T>
inline void distill3(std::array<T, 3>& a) {
    const auto [s0, e0] = Sorted ? fast_two_sum(a[0], a[1]) : two_sum(a[0], a[1]);
    const auto [s1, e1] = two_sum(s0, a[2]);
    const auto [b1, b2] = two_sum(e0, e1);

    // branchless conditional swap to ensure |a[0]| >= |a[1]|
    const bool swap = std::fabs(s1) < std::fabs(b1);
    const T hi = swap ? b1 : s1;
    const T lo = swap ? s1 : b1;
    a = { hi, lo, b2 };
}

// Sorted: if true, assumes |a[0]| >= |a[1]| and |a[2]| >= |a[3]| (uses fast_two_sum for both initial pairs)
template <bool Sorted = false, std::floating_point T>
inline void distill4(std::array<T, 4>& a) {
    const auto [s0, e0] = Sorted ? fast_two_sum(a[0], a[1]) : two_sum(a[0], a[1]);
    const auto [s1, e1] = Sorted ? fast_two_sum(a[2], a[3]) : two_sum(a[2], a[3]);
    const auto [s2, e2] = two_sum(s0, s1);
    const auto [t0, f0] = two_sum(e0, e1);
    const auto [t1, f1] = two_sum(t0, e2);

    // branchless conditional swap to ensure |a[2]| >= |a[3]|
    const bool swap = std::fabs(f1) < std::fabs(f0);
    const T lo = swap ? f1 : f0;
    const T hi = swap ? f0 : f1;

    // outputs are in sorted order: |s2| >= |t1| >= |hi| >= |lo|
    a = { s2, t1, hi, lo };
}

// Computes the error-free transformation of the sum of three numbers.
// Returns the rounded sum and the first-order error term.
template <std::floating_point T>
std::tuple<T, T> eft_add3(T x0, T x1, T x2) {
    // perform 2 rounds of distillation
    std::array<T, 3> a = { x0, x1, x2 };
    distill3<false>(a);
    distill3<true>(a);
    return { a[0], a[1] };
}

// Computes the error-free transformation of the sum of four numbers.
// Returns the rounded sum and the first-order error term.
template <std::floating_point T>
std::tuple<T, T> eft_add4(T x0, T x1, T x2, T x3) {
    // perform 3 rounds of distillation
    std::array<T, 4> a = { x0, x1, x2, x3 };
    distill4<false>(a);
    distill4<true>(a);
    distill4<true>(a);
    return { a[0], a[1] };
}

/// @brief Finalizes the rounding of an EFT result to round-to-odd.
/// Assumes `high` and `low` are both finite.
template <std::floating_point T>
inline T round_finalize(T high, T low) {
    MPFX_DEBUG_ASSERT(std::isfinite(high), "round_finalize: high part is not finite");
    MPFX_DEBUG_ASSERT(std::isfinite(low), "round_finalize: low part is not finite");
    using FP = typename mpfx::float_params<T>::params;
    using U = typename mpfx::float_params<T>::uint_t;
    static constexpr auto SIGN_SHIFT = FP::N - 1;

    // fast path: low part is zero
    if (low == static_cast<T>(0)) {
        return high; // exact result, no adjustment needed
    }

    // slow path: low part is non-zero (so is high part)
    const U b_high = std::bit_cast<U>(high);
    const U b_low = std::bit_cast<U>(low);

    // compute sign difference
    const int sign_high = b_high >> SIGN_SHIFT;
    const int sign_low = b_low >> SIGN_SHIFT;
    const int sign_diff = sign_high ^ sign_low;

    // compute adjustment for RTZ: +1 for negative `high`, -1 for positive `high`
    // only apply if the signs differ
    const int adjust_mask = -static_cast<int>(sign_diff);
    const int adjust = static_cast<int>((sign_high << 1) - 1) & adjust_mask;

    // apply adjustment and jam sticky bit for RTO
    U result = b_high + adjust;
    result |= 1;

    // reinterpret back to floating-point
    return std::bit_cast<T>(result);
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

    const mpfx::Context ctx(P, NMIN, MAX, mpfx::RM::RNE);

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

template <int P = 24>
double mx_dot_prod_ref(const std::vector<mx_block_t>& a_blocks, const std::vector<mx_block_t>& b_blocks) {
    MPFX_ASSERT(a_blocks.size() == b_blocks.size(), "block size mismatch");

    double result = 0.0;
    for (size_t i = 0; i < a_blocks.size(); i++) {
        // unpack blocks
        const auto [a_scale, a_elts] = a_blocks[i];
        const auto [b_scale, b_elts] = b_blocks[i];

        // multiply scales
        const double scale = a_scale + b_scale;

        // compute unscaled dot product
        double prod = 0.0;
        for (size_t j = 0; j < a_elts.size(); j++) {
            prod += a_elts[j] * b_elts[j];
        }

        // scale the product
        const double scaled = scale * prod;

        // add to result
        result += scaled;
    }

    return result;
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
            // unscaled dot product should be performed exactly
            mpfx::int128_t prod = 0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                const double p = a_elts[j] * b_elts[j];
                prod += mpfx::to_fixed(p, A_EXPMIN + B_EXPMIN);
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
            const auto [sum_hi, sum_lo] = eft_add3(scaled_hi, scaled_lo, result);
            result = mpfx::round(round_finalize(sum_hi, sum_lo), accum_ctx);
        } else if constexpr ((FA == MX::E5M2 && FB == MX::E4M3) || (FA == MX::E4M3 && FB == MX::E5M2)) {
            // unscaled dot product should be performed with at least 2*P bits of precision
            int64_t prod = 0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                const double p = a_elts[j] * b_elts[j];
                prod += mpfx::to_fixed(p, A_EXPMIN + B_EXPMIN);
            }

            // break up prod into 2 parts of 32 digits each
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
            const auto [sum_hi, sum_lo] = eft_add3(scaled_hi, scaled_lo, result);
            result = mpfx::round(round_finalize(sum_hi, sum_lo), accum_ctx);
        } else {
            // compute unscaled dot product using FP128
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
    {
        auto start = std::chrono::steady_clock::now();
        volatile double sf_result = 0.0;
        for (size_t i = 0; i < N; i++) {
            sf_result = mx_dot_prod_sf<FA, FB>(x_blocks[i], y_blocks[i]);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> sf_duration = end - start;
        sf_time = sf_duration.count();
        (void) sf_result; // suppress unused variable warning
    }

    // Time MPFX implementation
    double mpfx_time;
    {
        auto start = std::chrono::steady_clock::now();
        volatile double result = 0.0;
        for (size_t i = 0; i < x_blocks.size(); i++) {
            result = mx_dot_prod_impl<FA, FB>(x_blocks[i], y_blocks[i]);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;
        mpfx_time = duration.count();
        (void) result; // suppress unused variable warning
    }

    return { config, sf_time, mpfx_time };
}


int main(int argc, char* argv[]) {
    // Command line argument specifies number of inputs to test
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [num_inputs]" << std::endl;
        return 1;
    }

    const size_t N = std::stoul(argv[1]);

    // Configuration
    static constexpr size_t LEN = 4096;

    // Generate random test data
    std::random_device rd;
    std::mt19937_64 rng(rd());
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

    // Print table
    const int cw = 14; // column width
    std::cout << std::left << std::setw(cw) << "Config"
              << "| " << std::setw(cw) << "SoftFloat"
              << "| " << std::setw(cw) << "MPFX"
              << "| " << std::setw(cw) << "Speedup" << "|" << std::endl;
    std::cout << std::string(cw, '-')
              << "|" << std::string(cw + 1, '-')
              << "|" << std::string(cw + 1, '-')
              << "|" << std::string(cw + 1, '-') << "|" << std::endl;
    for (const auto& r : results) {
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

    return 0;
}
