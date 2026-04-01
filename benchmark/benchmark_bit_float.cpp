#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include <mpfx.hpp>

template <size_t N>
double benchmark_double(
    const std::vector<double>& x_vals,
    const std::vector<mpfx::prec_t>& prec_vals,
    const std::vector<mpfx::exp_t>& n_vals,
    mpfx::RM rm
) {
    const auto start = std::chrono::steady_clock::now();

    volatile double result = 0.0;
    for (size_t i = 0; i < N; i++) {
        result = mpfx::round(x_vals[i], prec_vals[i], n_vals[i], rm);
    }

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    return static_cast<double>(duration) / N;
}

template <size_t N, mpfx::RM RM>
double benchmark_bit_float(
    const std::vector<double>& x_vals,
    const std::vector<mpfx::prec_t>& prec_vals,
    const std::vector<mpfx::exp_t>& n_vals
) {
    const auto start = std::chrono::steady_clock::now();

    volatile double result = 0.0;
    for (size_t i = 0; i < N; i++) {
        const mpfx::bit_float<double> x(x_vals[i]);
        const auto y = mpfx::experimental::round<RM>(x, prec_vals[i], n_vals[i]);
        result = y.to_float();
    }

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    return static_cast<double>(duration) / N;
}


int main() {
    // configuration
    static constexpr size_t N = 100'000'000; // 100 million operations
    static constexpr mpfx::RM RM = mpfx::RM::RNE;
    static constexpr mpfx::prec_t PREC_MAX = 53;
    static constexpr mpfx::exp_t N_MAX = 0;
    static constexpr mpfx::exp_t N_MIN = -150;

    // input vector
    std::cout << "Allocating vectors...\n";

    std::vector<double> x_vals(N);
    std::vector<mpfx::prec_t> prec_vals(N);
    std::vector<mpfx::exp_t> n_vals(N);

    // generate random test data
    std::cout << "Generating random test data...\n";

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-1e+10, 1e+10);
    std::uniform_int_distribution<mpfx::prec_t> prec_dist(1, PREC_MAX);
    std::uniform_int_distribution<mpfx::exp_t> n_dist(N_MIN, N_MAX);

    for (size_t i = 0; i < N; i++) {
        x_vals[i] = dist(rng);
        prec_vals[i] = prec_dist(rng);
        n_vals[i] = n_dist(rng);
    }

    std::cout << "Timing mpfx::round(mpfx::bit_float<double>, ...)\n";
    const double avg_time_double = benchmark_double<N>(x_vals, prec_vals, n_vals, RM);

    std::cout << "Timing mpfx::round(mpfx::bit_float<double>, ...)\n";
    const double avg_time_bit_float = benchmark_bit_float<N, RM>(x_vals, prec_vals, n_vals);

    // output results
    std::cout << "================ MPFX Bit-Float Rounding Benchmark Results ================\n";
    std::cout << "Total operations:        " << N << "\n";
    std::cout << "Rounding mode:           " << static_cast<int>(RM) << "\n";
    std::cout << "Average time (double):   " << std::fixed << std::setprecision(2) << avg_time_double << " ns/op\n";
    std::cout << "Average time (bit_float): " << std::fixed << std::setprecision(2) << avg_time_bit_float << " ns/op\n";
    std::cout << "==========================================================================\n";

    return 0;
}
