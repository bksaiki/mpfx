#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>

TEST(TestEngine, TestSoftFloatEngineAdd) {
    static constexpr size_t N = 1000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const double y = dist(rng);

        const double z_ref = mpfx::engine_fp::add(x, y, 53);
        const double z = mpfx::engine_sf::add(x, y, 53);
        EXPECT_EQ(z_ref, z);
    }
}

TEST(TestEngine, TestEFTEngineAdd) {
    static constexpr size_t N = 1000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const double y = dist(rng);

        const double z_ref = mpfx::engine_fp::add(x, y, 53);
        const double z = mpfx::engine_eft::add(x, y, 53);
        EXPECT_EQ(z_ref, z);
    }
}
