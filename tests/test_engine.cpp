#include <bit>
#include <cstdint>
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

// `engine_fp` reads the hardware status flags, so only the operation itself may
// run between clearing and reading them. Operands are deliberately produced by
// inexact arithmetic here: if that arithmetic drifts into the window its rounding
// is read back as the operation's and exact results grow a sticky bit, and if the
// operation drifts out the flags are read before it runs and the sticky bit is
// lost. Both are scheduling artifacts, so this pins the resulting values.
TEST(TestEngine, TestFPEngineFlagIsolation) {
    volatile double vone = 1.0, vthree = 3.0, vtiny = 0x1p-60;
    const double third = vone / vthree; // inexact
    const double one = vone;

    // exact operations must come back untouched
    EXPECT_EQ(std::bit_cast<uint64_t>(mpfx::engine_fp::add(third, -third, 53)),
              std::bit_cast<uint64_t>(0.0));
    EXPECT_EQ(std::bit_cast<uint64_t>(mpfx::engine_fp::mul(third, vone, 53)),
              std::bit_cast<uint64_t>(third));

    // 1 + 2^-60 truncates to 1.0, whose LSB is clear, so the sticky bit must show
    EXPECT_EQ(std::bit_cast<uint64_t>(mpfx::engine_fp::add(one, vtiny, 53)),
              std::bit_cast<uint64_t>(1.0) | 1);

    // and either way the two engines must agree
    EXPECT_EQ(mpfx::engine_fp::add(third, -third, 53), mpfx::engine_eft::add(third, -third, 53));
    EXPECT_EQ(mpfx::engine_fp::add(one, vtiny, 53), mpfx::engine_eft::add(one, vtiny, 53));
    EXPECT_EQ(mpfx::engine_fp::mul(third, vthree, 53), mpfx::engine_eft::mul(third, vthree, 53));
    EXPECT_EQ(mpfx::engine_fp::div(one, vthree, 53), mpfx::engine_eft::div(one, vthree, 53));
}
