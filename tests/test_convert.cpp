#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

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

TEST(TestConvert, TestToFixedInt64) {
    static_assert(std::is_same_v<decltype(mpfx::to_fixed(1.0, 0)), int64_t>);

    EXPECT_EQ(mpfx::to_fixed(0.0, -10), 0);
    EXPECT_EQ(mpfx::to_fixed(-0.0, -10), 0);
    EXPECT_EQ(mpfx::to_fixed(0.75, -2), 3);
    EXPECT_EQ(mpfx::to_fixed(-0.75, -2), -3);
    EXPECT_EQ(mpfx::to_fixed(96.0, 5), 3);

    // extremes of int64_t that a double represents exactly
    EXPECT_EQ(mpfx::to_fixed(0x1.fffffffffffffp62, 0), INT64_C(0x7ffffffffffffc00));
    EXPECT_EQ(mpfx::to_fixed(-0x1p63, 0), std::numeric_limits<int64_t>::min());
}

TEST(TestConvert, TestToFixedInt128) {
    using mpfx::int128_t;
    static_assert(std::is_same_v<decltype(mpfx::to_fixed<int128_t>(1.0, 0)), int128_t>);

    EXPECT_TRUE(mpfx::to_fixed<int128_t>(0.0, -10) == 0);
    EXPECT_TRUE(mpfx::to_fixed<int128_t>(0.75, -2) == 3);
    EXPECT_TRUE(mpfx::to_fixed<int128_t>(-0.75, -2) == -3);

    // E5M2 max squared (49 * 2^26) at quantum 2^-32: 49 * 2^58 > INT64_MAX
    const double e5m2_sq = 57344.0 * 57344.0;
    const int128_t expected = static_cast<int128_t>(49) << 58;
    EXPECT_TRUE(expected > std::numeric_limits<int64_t>::max());
    EXPECT_TRUE(mpfx::to_fixed<int128_t>(e5m2_sq, -32) == expected);
    EXPECT_TRUE(mpfx::to_fixed<int128_t>(-e5m2_sq, -32) == -expected);

    // beyond the uint64_t range
    EXPECT_TRUE(mpfx::to_fixed<int128_t>(0x1.8p100, 0) == (static_cast<int128_t>(3) << 99));
    EXPECT_TRUE(mpfx::to_fixed<int128_t>(-0x1.8p100, 0) == -(static_cast<int128_t>(3) << 99));
}
