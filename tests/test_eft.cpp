#include <array>
#include <random>

#include <gtest/gtest.h>
#include <mpfx/engine_eft.hpp>
#include <mpfx.hpp>

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
        const auto [s0, s1, s2] = mpfx::engine_eft::eft_add3(x0, x1, x2);
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
        const auto [s0, s1, s2, s3] = mpfx::engine_eft::eft_add4(x0, x1, x2, x3);
        EXPECT_TRUE(nonoverlapping_check(s0, s1));
        EXPECT_TRUE(nonoverlapping_check(s1, s2));
        EXPECT_TRUE(nonoverlapping_check(s2, s3));
    }

}
