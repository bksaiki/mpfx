// Minimal standalone replica of the failing loop in tests/test_engine.cpp:
// nothing else in the loop body, so the generated code can be read directly and
// nothing accidentally pins the operation inside the RTO window.

#include <cstdint>
#include <cstdio>
#include <random>

#include <mpfx/arch.hpp>
#include <mpfx/engine_eft.hpp>
#include <mpfx/engine_fp.hpp>

// noinline so the loop is easy to locate in -S output
__attribute__((noinline))
static size_t run(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = dist(rng);
        const double y = dist(rng);
        const double z_ref = mpfx::engine_fp::add(x, y, 53);
        const double z = mpfx::engine_eft::add(x, y, 53);
        if (z_ref != z) bad++;
    }
    return bad;
}

// Same loop, but the results are consumed by const reference, the way gtest's
// EXPECT_EQ takes them. That forces both values into memory and may be what
// changes the scheduling in the real test.
__attribute__((noinline))
static void compare_by_ref(const double& a, const double& b, size_t& bad) {
    if (a != b) bad++;
}

__attribute__((noinline))
static size_t run_byref(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = dist(rng);
        const double y = dist(rng);
        const double z_ref = mpfx::engine_fp::add(x, y, 53);
        const double z = mpfx::engine_eft::add(x, y, 53);
        compare_by_ref(z_ref, z, bad);
    }
    return bad;
}

int main() {
    constexpr size_t N = 1000000;
    std::printf("entry: fpscr=0x%08x rounding=%d\n",
                (unsigned) mpfx::arch::get_fpscr(), mpfx::arch::get_rounding_mode());
    const size_t bad = run(N, 20260813);
    std::printf("mismatches (by value): %zu / %zu\n", bad, N);
    std::printf("mismatches (by ref):   %zu / %zu\n", run_byref(N, 20260813), N);
    std::printf("exit:  fpscr=0x%08x rounding=%d\n",
                (unsigned) mpfx::arch::get_fpscr(), mpfx::arch::get_rounding_mode());
    return 0;
}
