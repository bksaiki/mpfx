#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include <mpfr.h>
#include <mpfx.hpp>

template <mpfx::flag_mask_t FlagMask = mpfx::Flags::ALL_FLAGS>
double run_benchmark(const std::vector<double>& x_vals, const mpfx::Context& ctx, size_t N) {
    std::cout << "Starting MPFX rounding benchmark...\n";
    auto start = std::chrono::steady_clock::now();

    volatile double result;
    for (const double x : x_vals) {
        result = mpfx::round<FlagMask>(x, ctx);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning

    return static_cast<double>(duration) / N; // average time per operation in ns
}


int main() {
    // configuration
    static constexpr size_t N = 100'000'000; // 100 million operations
    static constexpr size_t ES = 8;
    static constexpr size_t NBITS = 32;
    static constexpr mpfx::RM RM = mpfx::RM::RNE;

    // create MPFX IEEE 754 context
    const mpfx::IEEE754Context ctx(ES, NBITS, RM);

    // input vector
    std::vector<double> x_vals(N);

    // generate random test data
    std::cout << "Generating random test data...\n";

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-1e+10, 1e+10);
    for (size_t i = 0; i < N; i++) {
        x_vals[i] = dist(rng);
    }

    double avg_time = run_benchmark<mpfx::Flags::ALL_FLAGS>(x_vals, ctx, N);
    double avg_time_no_flags = run_benchmark<mpfx::Flags::NO_FLAGS>(x_vals, ctx, N);

    std::cout << "MPFX rounding benchmark completed.\n\n";

    // output results
    std::cout << "================ MPFX Rounding Benchmark Results ================\n";
    std::cout << "Total operations:        " << N << "\n";
    std::cout << "Precision:               " << static_cast<size_t>(ctx.prec()) << " bits\n";
    std::cout << "Rounding mode:           " << static_cast<int>(ctx.rm()) << "\n";
    std::cout << "Average time:            " << std::fixed << std::setprecision(2) << avg_time << " ns/op\n";
    std::cout << "Average time (no flags): " << std::fixed << std::setprecision(2) << avg_time_no_flags << " ns/op\n";
    std::cout << "===============================================================\n";

    return 0;
}
