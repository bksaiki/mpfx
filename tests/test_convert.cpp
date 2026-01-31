#include <limits>
#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>

TEST(TestConvert, TestUnpackFloatFP64) {
    using limits = std::numeric_limits<double>;
    using FP = mpfx::float_params<double>::params;

    // +0.0
    {
        const double x = 0.0;
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, 0);
    }

    // -0.0
    {
        const double x = -0.0;
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, true);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, 0);
    }

    // +(minimum subnormal)
    {
        const double x = limits::denorm_min();
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, 1);
    }

    // +(maximum subnormal)
    {
        const double x = limits::denorm_min() * static_cast<double>(FP::IMPLICIT1 - 1);
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, FP::IMPLICIT1 - 1);
    }

    // +(minimum normal)
    {
        const double x = limits::denorm_min() * static_cast<double>(FP::IMPLICIT1);
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, FP::IMPLICIT1);
    }

    // +1.0
    {
        const double x = 1.0;
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, -static_cast<mpfx::exp_t>(FP::M));
        EXPECT_EQ(c, FP::IMPLICIT1);
    }

    // +(maximum normal)
    {
        const double x = limits::max();
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMAX);
        EXPECT_EQ(c, FP::IMPLICIT1 | FP::MMASK);
    }
}

TEST(TestConvert, TestUnpackFloatFP32) {
    using limits = std::numeric_limits<float>;
    using FP = mpfx::float_params<float>::params;

    // +0.0
    {
        const float x = 0.0f;
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, 0);
    }

    // -0.0
    {
        const float x = -0.0f;
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, true);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, 0);
    }

    // +(minimum subnormal)
    {
        const float x = limits::denorm_min();
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, 1);
    }

    // +(maximum subnormal)
    {
        const float x = limits::denorm_min() * static_cast<float>(FP::IMPLICIT1 - 1);
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, FP::IMPLICIT1 - 1);
    }

    // +(minimum normal)
    {
        const float x = limits::denorm_min() * static_cast<float>(FP::IMPLICIT1);
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMIN);
        EXPECT_EQ(c, FP::IMPLICIT1);
    }

    // +1.0
    {
        const float x = 1.0f;
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, -static_cast<mpfx::exp_t>(FP::M));
        EXPECT_EQ(c, FP::IMPLICIT1);
    }

    // +(maximum normal)
    {
        const float x = limits::max();
        const auto [s, exp, c] = mpfx::unpack_float(x);
        EXPECT_EQ(s, false);
        EXPECT_EQ(exp, FP::EXPMAX);
        EXPECT_EQ(c, FP::IMPLICIT1 | FP::MMASK);
    }
}

TEST(TestConvert, TestUnpackPackFP64) {
    using FP = mpfx::float_params<double>::params;
    using limits = mpfx::float_params<double>::limits;
    using uint_t = mpfx::float_params<double>::uint_t;
    using int_t = std::make_signed_t<uint_t>;
    static constexpr auto MAX_VAL = limits::max();
    static constexpr auto MAX_ORD = std::bit_cast<uint_t>(MAX_VAL);
    static constexpr size_t N = 1'000'000;

    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_int_distribution<int_t> dist(-MAX_ORD, MAX_ORD);

    for (size_t i = 0; i < N; i++) {
        const int_t ord = dist(rng);
        const uint_t bits = (ord < 0) ? (static_cast<uint_t>(-ord) | FP::SMASK) : static_cast<uint_t>(ord);
        const double x = std::bit_cast<double>(static_cast<uint_t>(bits));
        const auto [s, exp, c] = mpfx::unpack_float(x);
        const double y = mpfx::make_float<double>(s, exp, c);
        EXPECT_EQ(x, y);
    }
}

TEST(TestConvert, TestUnpackPackFP32) {
    using FP = mpfx::float_params<float>::params;
    using limits = mpfx::float_params<float>::limits;
    using uint_t = mpfx::float_params<float>::uint_t;
    using int_t = std::make_signed_t<uint_t>;
    static constexpr auto MAX_VAL = limits::max();
    static constexpr auto MAX_ORD = std::bit_cast<uint_t>(MAX_VAL);
    static constexpr size_t N = 1'000'000;

    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_int_distribution<int_t> dist(-MAX_ORD, MAX_ORD);

    for (size_t i = 0; i < N; i++) {
        const int_t ord = dist(rng);
        const uint_t bits = (ord < 0) ? (static_cast<uint_t>(-ord) | FP::SMASK) : static_cast<uint_t>(ord);
        const float x = std::bit_cast<float>(static_cast<uint_t>(bits));
        const auto [s, exp, c] = mpfx::unpack_float(x);
        const float y = mpfx::make_float<float>(s, exp, c);
        EXPECT_EQ(x, y);
    }
}
