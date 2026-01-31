#include <limits>

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
