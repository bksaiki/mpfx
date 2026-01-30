#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>

TEST(TestEngine, TestEFTEngineAdd) {
    static constexpr size_t N = 10000000;

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

TEST(TestEngine, TestEFTEngineSub) {
    static constexpr size_t N = 10000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const double y = dist(rng);

        const double z_ref = mpfx::engine_fp::sub(x, y, 53);
        const double z = mpfx::engine_eft::sub(x, y, 53);
        EXPECT_EQ(z_ref, z);
    }
}

TEST(TestEngine, TestEFTEngineMul) {
    static constexpr size_t N = 10000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const double y = dist(rng);

        const double z_ref = mpfx::engine_fp::mul(x, y, 53);
        const double z = mpfx::engine_eft::mul(x, y, 53);
        EXPECT_EQ(z_ref, z);
    }
}

TEST(TestEngine, TestEFTEngineDiv) {
    static constexpr size_t N = 10000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const double y = dist(rng);

        const double z_ref = mpfx::engine_fp::div(x, y, 53);
        const double z = mpfx::engine_eft::div(x, y, 53);
        EXPECT_EQ(z_ref, z);
    }
}

TEST(TestEngine, TestEFTEngineSqrt) {
    static constexpr size_t N = 10000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);

        const double z_ref = mpfx::engine_fp::sqrt(x, 53);
        const double z = mpfx::engine_eft::sqrt(x, 53);
        EXPECT_EQ(z_ref, z);
    }
}

TEST(TestEngine, TestEFTEngineFma) {
    static constexpr size_t N = 10000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const double y = dist(rng);
        const double z = dist(rng);

        const double w_ref = mpfx::engine_fp::fma(x, y, z, 53);
        const double w = mpfx::engine_eft::fma(x, y, z, 53);
        EXPECT_EQ(w_ref, w);
    }
}
