#include <array>
#include <random>

#include <gtest/gtest.h>
#include <mpfx.hpp>

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
    a = { s2, t1, hi, lo };
}

template <std::floating_point T>
std::array<T, 3> eft_add3(T x0, T x1, T x2) {
    // perform 2 rounds of distillation
    std::array<T, 3> a = { x0, x1, x2 };
    distill3<false>(a);
    distill3<true>(a);
    return a;
}

// Computes the error-free transformation of the sum of four numbers.
// Returns the rounded sum and the first-order error term.
template <std::floating_point T>
std::array<T, 4> eft_add4(T x0, T x1, T x2, T x3) {
    // perform "distillation" according to Priest
    std::array<T, 4> a = { x0, x1, x2, x3 };
    distill4<false>(a);
    distill4<true>(a);
    distill4<true>(a);
    return a;
}


inline bool nonoverlapping_check(double x, double y) {
    if (x == 0.0) {
        return y == 0.0;
    } else if (y == 0.0) {
        return true;
    } else {
        // unpack floating-point values
        const auto xparts = mpfx::unpack_float(x);
        const auto yparts = mpfx::unpack_float(y);
        const auto ex = std::get<1>(xparts);
        const auto ey = std::get<1>(yparts);
        return ex - ey >= 53;
    }
}

TEST(TestEFT, TestEFTAdd3) {
    static constexpr size_t N = 1'000'000;

    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> exp_dist(-150, 150);

    for (size_t i = 0; i < N; i++) {
        const double x0 = std::ldexp(dist(rng), exp_dist(rng));
        const double x1 = std::ldexp(dist(rng), exp_dist(rng));
        const double x2 = std::ldexp(dist(rng), exp_dist(rng));

        // std::cout << "Test case " << i << ": " << x0 << ", " << x1 << ", " << x2 << ", " << x3 << std::endl;
        const auto [s0, s1, s2] = eft_add3(x0, x1, x2);
        EXPECT_TRUE(nonoverlapping_check(s0, s1));
        EXPECT_TRUE(nonoverlapping_check(s1, s2));
    }

}


TEST(TestEFT, TestEFTAdd4) {
    static constexpr size_t N = 1'000'000;

    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> exp_dist(-150, 150);

    for (size_t i = 0; i < N; i++) {
        const double x0 = std::ldexp(dist(rng), exp_dist(rng));
        const double x1 = std::ldexp(dist(rng), exp_dist(rng));
        const double x2 = std::ldexp(dist(rng), exp_dist(rng));
        const double x3 = std::ldexp(dist(rng), exp_dist(rng));

        // std::cout << "Test case " << i << ": " << x0 << ", " << x1 << ", " << x2 << ", " << x3 << std::endl;
        const auto [s0, s1, s2, s3] = eft_add4(x0, x1, x2, x3);
        EXPECT_TRUE(nonoverlapping_check(s0, s1));
        EXPECT_TRUE(nonoverlapping_check(s1, s2));
        EXPECT_TRUE(nonoverlapping_check(s2, s3));
    }

}
