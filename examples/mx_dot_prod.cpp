#include <array>
#include <chrono>
#include <cmath>
#include <random>
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


using mx_block_t = std::tuple<int, std::array<double, 32>>;


template <size_t K, MX F, size_t N>
std::vector<mx_block_t> mx_block_quantize(const std::array<double, N>& vec) {
    
    // number of blocks (last block may be smaller)
    constexpr size_t num_blocks = (N + K - 1) / K;

    // rounding context
    constexpr int P = mx_params<F>::P;
    constexpr int NMIN = mx_params<F>::NMIN;
    constexpr int MAX = mx_params<F>::MAX;
    constexpr int EMAX = std::ilogb(MAX);
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
        const int scale = max_e - EMAX;

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
        const int scale = a_scale + b_scale;

        // compute unscaled dot product
        double prod = 0.0;
        for (size_t j = 0; j < a_elts.size(); j++) {
            prod += a_elts[j] * b_elts[j];
        }

        // scale the product
        const double scaled = std::ldexp(prod, scale);

        // add to result
        result += scaled;
    }

    return result;
}

template <int P = 24>
double mx_dot_prod_sf(const std::vector<mx_block_t>& a_blocks, const std::vector<mx_block_t>& b_blocks) {
    MPFX_ASSERT(a_blocks.size() == b_blocks.size(), "block size mismatch");

    float32_t result = { 0 };
    for (size_t i = 0; i < a_blocks.size(); i++) {
        // unpack blocks
        const auto [a_scale, a_elts] = a_blocks[i];
        const auto [b_scale, b_elts] = b_blocks[i];

        // multiply scales
        float128_t scale;
        f64_to_f128M(std::bit_cast<float64_t>(std::ldexp(1.0, a_scale + b_scale)), &scale);

        // compute unscaled dot product using FP128
        float128_t prod = { 0 };
        for (size_t j = 0; j < a_elts.size(); j++) {
            float64_t a_f64 = std::bit_cast<float64_t>(a_elts[j]);
            float64_t b_f64 = std::bit_cast<float64_t>(b_elts[j]);

            float128_t a, b, p;
            f64_to_f128M(a_f64, &a);
            f64_to_f128M(b_f64, &b);
            f128M_mul(&a, &b, &p);
            f128M_add(&prod, &p, &prod);
        }

        // scale the product
        f128M_mul(&prod, &scale, &prod);

        // round result to P bits of precision with scaling
        softfloat_roundingMode = softfloat_round_minMag;
        const float32_t scaled = f128M_to_f32(&prod);

        // add to result
        softfloat_roundingMode = softfloat_round_minMag;
        result = f32_add(result, scaled);
    }

    return static_cast<double>(std::bit_cast<float>(result));
}

template <MX FA, MX FB>
double mx_dot_prod_impl(const std::vector<mx_block_t>& a_blocks, const std::vector<mx_block_t>& b_blocks) {
    constexpr auto A_EXPMIN = mx_params<FA>::NMIN + 1;
    constexpr auto B_EXPMIN = mx_params<FB>::NMIN + 1;
    MPFX_ASSERT(a_blocks.size() == b_blocks.size(), "block size mismatch");

    // rounding contexts
    const mpfx::IEEE754Context scale_ctx(8, 32, mpfx::RM::RTZ);
    const mpfx::IEEE754Context accum_ctx(8, 32, mpfx::RM::RTZ);

    double result = 0.0;
    for (size_t i = 0; i < a_blocks.size(); i++) {
        // unpack blocks
        const auto [a_scale, a_elts] = a_blocks[i];
        const auto [b_scale, b_elts] = b_blocks[i];

        // multiply scales
        const int scale = a_scale + b_scale;

        // compute scaled dot product
        double scaled;
        if constexpr (FA == MX::E5M2 && FB == MX::E5M2) {
            // unscaled dot product should be performed exactly
            mpfx::int128_t prod = 0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                const mpfx::int128_t a = mpfx::to_fixed(a_elts[j], A_EXPMIN);
                const mpfx::int128_t b = mpfx::to_fixed(b_elts[j], B_EXPMIN);
                prod += a * b;
            }

            // round to P bits of precision with scaling
            scaled = scale_ctx.round(prod, scale + A_EXPMIN + B_EXPMIN);
        } else if constexpr ((FA == MX::E5M2 && FB == MX::E4M3) || (FA == MX::E4M3 && FB == MX::E5M2)) {
            // unscaled dot product should be performed with at least 2*P bits of precision
            int64_t prod = 0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                const int64_t a = mpfx::to_fixed(a_elts[j], A_EXPMIN);
                const int64_t b = mpfx::to_fixed(b_elts[j], B_EXPMIN);
                prod += a * b;
            }

            // round to P bits of precision with scaling
            scaled = scale_ctx.round(prod, scale + A_EXPMIN + B_EXPMIN);
        } else {
            // compute unscaled dot product using FP128
            double prod = 0.0;
            for (size_t j = 0; j < a_elts.size(); j++) {
                prod += a_elts[j] * b_elts[j];
            }

            // scale the product
            scaled = std::ldexp(prod, scale);
        }

        // add to result with rounding
        result = mpfx::add<mpfx::Engine::EFT>(result, scaled, accum_ctx);
    }

    return result;
}

template <MX FA, MX FB, size_t LEN>
void time_dot_prod(
    const std::vector<std::array<double, LEN>>& x_vals,
    const std::vector<std::array<double, LEN>>& y_vals
) {
    const size_t N = x_vals.size();

    // Quantize data into blocks
    std::vector<std::vector<mx_block_t>> x_blocks(N);
    std::vector<std::vector<mx_block_t>> y_blocks(N);
    for (size_t i = 0; i < N; i++) {
        x_blocks[i] = mx_block_quantize<32, FA>(x_vals[i]);
        y_blocks[i] = mx_block_quantize<32, FB>(y_vals[i]);
    } 

    // Time softfloat reference implementation
    {
        auto start = std::chrono::steady_clock::now();
        volatile double sf_result = 0.0;
        for (size_t i = 0; i < N; i++) {
            sf_result = mx_dot_prod_sf(x_blocks[i], y_blocks[i]);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> sf_duration = end - start;
        std::cout << "SoftFloat time: " << sf_duration.count() << " seconds" << std::endl;
        (void) sf_result; // suppress unused variable warning
    }

    // Time MPFX implementation
    {
        auto start = std::chrono::steady_clock::now();
        volatile double result = 0.0;
        for (size_t i = 0; i < x_blocks.size(); i++) {
            result = mx_dot_prod_impl<FA, FB>(x_blocks[i], y_blocks[i]);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "MPFX time:      " << duration.count() << " seconds" << std::endl;
        (void) result; // suppress unused variable warning
    }
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
    time_dot_prod<MX::E5M2, MX::E5M2>(x_vals, y_vals);
    time_dot_prod<MX::E5M2, MX::E4M3>(x_vals, y_vals);
    time_dot_prod<MX::E4M3, MX::E4M3>(x_vals, y_vals);
    time_dot_prod<MX::E2M3, MX::E2M3>(x_vals, y_vals);
    time_dot_prod<MX::E2M1, MX::E2M1>(x_vals, y_vals);

    return 0;
}
