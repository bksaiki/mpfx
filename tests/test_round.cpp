#include <bit>
#include <climits>
#include <optional>
#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>

using namespace mpfx;

using round_test_t = std::tuple<exp_t, mant_t, exp_t, mant_t, mpfx::RM>;

TEST(TestRound, TestRoundExamples) {
    EXPECT_EQ(round(0.0, 1, std::nullopt, RM::RNE), 0.0);
    EXPECT_EQ(round(std::bit_cast<double>(1ULL), 1, std::nullopt, RM::RNE), std::bit_cast<double>(1ULL));
    EXPECT_EQ(round(std::bit_cast<double>(3ULL), 1, std::nullopt, RM::RTZ), std::bit_cast<double>(2ULL));

    EXPECT_EQ(round(0.75, 8, -1, RM::RNE), 1.0);
    EXPECT_EQ(round(0.75, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(0.75, 8, -1, RM::RTZ), 0.0);

    EXPECT_EQ(round(0.5, 8, -1, RM::RNE), 0.0);
    EXPECT_EQ(round(0.5, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(0.5, 8, -1, RM::RTZ), 0.0);
    
    EXPECT_EQ(round(0.25, 8, -1, RM::RNE), 0.0);
    EXPECT_EQ(round(0.25, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(0.25, 8, -1, RM::RTZ), 0.0);
}

TEST(TestRound, TestRoundFixedExamples) {
    EXPECT_EQ(round(0, 50, 1, std::nullopt, RM::RNE), 0.0);
    EXPECT_EQ(round(1, 0, 1, std::nullopt, RM::RNE), 1.0);
    EXPECT_EQ(round(3, 0, 1, std::nullopt, RM::RTZ), 2.0);
    EXPECT_EQ(round(-1, 0, 1, std::nullopt, RM::RNE), -1.0);
    EXPECT_EQ(round(-3, 0, 1, std::nullopt, RM::RTZ), -2.0);

    EXPECT_EQ(round(3, -2, 8, -1, RM::RNE), 1.0);
    EXPECT_EQ(round(3, -2, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(3, -2, 8, -1, RM::RTZ), 0.0);

    EXPECT_EQ(round(2, -2, 8, -1, RM::RNE), 0.0);
    EXPECT_EQ(round(2, -2, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(2, -2, 8, -1, RM::RTZ), 0.0);

    EXPECT_EQ(round(1, -2, 8, -1, RM::RNE), 0.0);
    EXPECT_EQ(round(1, -2, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(1, -2, 8, -1, RM::RTZ), 0.0);
}

TEST(TestRound, TestRoundFixed128Examples) {
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(0), 50, 1, std::nullopt, RM::RNE), 0.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(1), 0, 1, std::nullopt, RM::RNE), 1.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(3), 0, 1, std::nullopt, RM::RTZ), 2.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(-1), 0, 1, std::nullopt, RM::RNE), -1.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(-3), 0, 1, std::nullopt, RM::RTZ), -2.0);

    EXPECT_EQ(round(static_cast<mpfx::int128_t>(3), -2, 8, -1, RM::RNE), 1.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(3), -2, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(3), -2, 8, -1, RM::RTZ), 0.0);

    EXPECT_EQ(round(static_cast<mpfx::int128_t>(2), -2, 8, -1, RM::RNE), 0.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(2), -2, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(2), -2, 8, -1, RM::RTZ), 0.0);

    EXPECT_EQ(round(static_cast<mpfx::int128_t>(1), -2, 8, -1, RM::RNE), 0.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(1), -2, 8, -1, RM::RAZ), 1.0);
    EXPECT_EQ(round(static_cast<mpfx::int128_t>(1), -2, 8, -1, RM::RTZ), 0.0);
}


TEST(TestRound, TestRoundWithPrec) {
    std::vector<round_test_t> inputs = {
        // 8 * 2 ** -3 (representable)
        {-3, 8, -1, 2, RM::RNE}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RNA}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTP}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTN}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RAZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        // 9 * 2 ** -3 (below halfway)
        {-3, 9, -1, 2, RM::RNE}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RNA}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RTP}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 9, -1, 2, RM::RTN}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RTZ}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RAZ}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 10 * 2 ** -3 (exactly halfway)
        {-3, 10, -1, 2, RM::RNE}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RNA}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 3, RM::RTP}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 2, RM::RTN}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 2, RM::RTZ}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RAZ}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 11 * 2 ** -3 (above halfway)
        {-3, 11, -1, 3, RM::RNE}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RNA}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RTP}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 2, RM::RTN}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 2, RM::RTZ}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 3, RM::RAZ}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 12 * 2 ** -3 (representable)
        {-3, 12, -1, 3, RM::RNE}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RNA}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTP}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTN}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RAZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
    };

    for (const auto& [exp_in, c_in, exp_out, c_out, rm] : inputs) {
        const auto x = static_cast<double>(mpfx::make_float<double>(false, exp_in, c_in));
        const auto y_expect = static_cast<double>(mpfx::make_float<double>(false, exp_out, c_out));
        const auto y = round(x, 2, std::nullopt, rm);
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundWithPrecFixed) {
    std::vector<round_test_t> inputs = {
        // 8 * 2 ** -3 (representable)
        {-3, 8, -1, 2, RM::RNE}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RNA}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTP}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTN}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RAZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        // 9 * 2 ** -3 (below halfway)
        {-3, 9, -1, 2, RM::RNE}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RNA}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RTP}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 9, -1, 2, RM::RTN}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RTZ}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RAZ}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 10 * 2 ** -3 (exactly halfway)
        {-3, 10, -1, 2, RM::RNE}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RNA}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 3, RM::RTP}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 2, RM::RTN}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 2, RM::RTZ}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RAZ}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 11 * 2 ** -3 (above halfway)
        {-3, 11, -1, 3, RM::RNE}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RNA}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RTP}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 2, RM::RTN}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 2, RM::RTZ}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 3, RM::RAZ}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 12 * 2 ** -3 (representable)
        {-3, 12, -1, 3, RM::RNE}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RNA}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTP}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTN}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RAZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
    };

    for (const auto& [exp_in, c_in, exp_out, c_out, rm] : inputs) {
        const auto y_expect = static_cast<double>(mpfx::make_float<double>(false, exp_out, c_out));
        const auto m_in = static_cast<int64_t>(c_in);
        const auto y = round(m_in, exp_in, 2, std::nullopt, rm);
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundWithN) {
    std::vector<round_test_t> inputs = {
        // 8 * 2 ** -3 (representable)
        {-3, 8, -1, 2, RM::RNE}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RNA}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTP}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTN}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RAZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        // 9 * 2 ** -3 (below halfway)
        {-3, 9, -1, 2, RM::RNE}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RNA}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RTP}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 9, -1, 2, RM::RTN}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RTZ}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RAZ}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 10 * 2 ** -3 (exactly halfway)
        {-3, 10, -1, 2, RM::RNE}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RNA}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 3, RM::RTP}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 2, RM::RTN}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 2, RM::RTZ}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RAZ}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 11 * 2 ** -3 (above halfway)
        {-3, 11, -1, 3, RM::RNE}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RNA}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RTP}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 2, RM::RTN}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 2, RM::RTZ}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 3, RM::RAZ}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 12 * 2 ** -3 (representable)
        {-3, 12, -1, 3, RM::RNE}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RNA}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTP}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTN}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RAZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
    };

    for (const auto& [exp_in, c_in, exp_out, c_out, rm] : inputs) {
        const auto x = static_cast<double>(mpfx::make_float<double>(false, exp_in, c_in));
        const auto y_expect = static_cast<double>(mpfx::make_float<double>(false, exp_out, c_out));
        const auto y = round(x, 3, -2, rm);
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundWithNFixed) {
    std::vector<round_test_t> inputs = {
        // 8 * 2 ** -3 (representable)
        {-3, 8, -1, 2, RM::RNE}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RNA}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTP}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTN}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RAZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        // 9 * 2 ** -3 (below halfway)
        {-3, 9, -1, 2, RM::RNE}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RNA}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RTP}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 9, -1, 2, RM::RTN}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RTZ}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RAZ}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 10 * 2 ** -3 (exactly halfway)
        {-3, 10, -1, 2, RM::RNE}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RNA}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 3, RM::RTP}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 2, RM::RTN}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 2, RM::RTZ}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RAZ}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 11 * 2 ** -3 (above halfway)
        {-3, 11, -1, 3, RM::RNE}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RNA}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RTP}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 2, RM::RTN}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 2, RM::RTZ}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 3, RM::RAZ}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 12 * 2 ** -3 (representable)
        {-3, 12, -1, 3, RM::RNE}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RNA}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTP}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTN}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RAZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
    };

    for (const auto& [exp_in, c_in, exp_out, c_out, rm] : inputs) {
        const auto y_expect = static_cast<double>(mpfx::make_float<double>(false, exp_out, c_out));
        const auto m_in = static_cast<int64_t>(c_in);
        const auto y = round(m_in, exp_in, 3, -2, rm);
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundBitFloat) {
    EXPECT_EQ(
        experimental::round<RM::RNE>(bit_float<float>(0.0f), 1, std::nullopt).to_float(),
        bit_float<float>(0.0f).to_float()
    );
    EXPECT_EQ(
        experimental::round<RM::RNE>(bit_float<float>(1.0f), 1, std::nullopt).to_float(),
        bit_float<float>(1.0f).to_float()
    );
    EXPECT_EQ(
        experimental::round<RM::RTZ>(bit_float<float>(3.0f), 1, std::nullopt).to_float(),
        bit_float<float>(2.0f).to_float()
    );
}

TEST(TestRound, TestRoundBitFloatCompareRNE) {
    static constexpr size_t N = 1'000'000;
    static constexpr RM rm = RM::RNE;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_int_distribution<uint32_t> bits_dist(0, 0xffffffff);
    std::uniform_int_distribution<prec_t> prec_dist(1, 24);
    std::uniform_int_distribution<exp_t> n_dist(-149, 0);

    for (size_t i = 0; i < N; i++) {
        // generate a random bitstring
        uint32_t raw_bits = bits_dist(rng);
        bit_float<float> bf(raw_bits);
        float v = bf.to_float();

        // skip NaN
        if (std::isnan(v)) {
            continue;
        }

        // generate a precision [1, 24]
        prec_t prec = prec_dist(rng);

        // generate a subnormalization point [-149, 0]
        exp_t n = n_dist(rng);

        // round via both methods and compare results
        float y_expect = round(v, prec, n, rm);
        float y = experimental::round<rm>(bf, prec, n).to_float();
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundBitFloatCompareRTZ) {
    static constexpr size_t N = 1'000'000;
    static constexpr RM rm = RM::RTZ;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_int_distribution<uint32_t> bits_dist(0, 0xffffffff);
    std::uniform_int_distribution<prec_t> prec_dist(1, 24);
    std::uniform_int_distribution<exp_t> n_dist(-149, 0);

    for (size_t i = 0; i < N; i++) {
        // generate a random bitstring
        uint32_t raw_bits = bits_dist(rng);
        bit_float<float> bf(raw_bits);
        float v = bf.to_float();

        // skip NaN
        if (std::isnan(v)) {
            continue;
        }

        // generate a precision [1, 24]
        prec_t prec = prec_dist(rng);

        // generate a subnormalization point [-149, 0]
        exp_t n = n_dist(rng);

        // round via both methods and compare results
        float y_expect = round(v, prec, n, rm);
        float y = experimental::round<rm>(bf, prec, n).to_float();
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundBitFloatCompareRAZ) {
    static constexpr size_t N = 1'000'000;
    static constexpr RM rm = RM::RAZ;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_int_distribution<uint32_t> bits_dist(0, 0xffffffff);
    std::uniform_int_distribution<prec_t> prec_dist(1, 24);
    std::uniform_int_distribution<exp_t> n_dist(-149, 0);

    for (size_t i = 0; i < N; i++) {
        // generate a random bitstring
        uint32_t raw_bits = bits_dist(rng);
        bit_float<float> bf(raw_bits);
        float v = bf.to_float();

        // skip NaN
        if (std::isnan(v)) {
            continue;
        }

        // generate a precision [1, 24]
        prec_t prec = prec_dist(rng);

        // generate a subnormalization point [-149, 0]
        exp_t n = n_dist(rng);

        // round via both methods and compare results
        float y_expect = round(v, prec, n, rm);
        float y = experimental::round<rm>(bf, prec, n).to_float();
        EXPECT_EQ(y, y_expect);
    }
}

TEST(TestRound, TestRoundBitFloatCompareRTO) {
    static constexpr size_t N = 1'000'000;
    static constexpr RM rm = RM::RTO;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    std::uniform_int_distribution<uint32_t> bits_dist(0, 0xffffffff);
    std::uniform_int_distribution<prec_t> prec_dist(1, 24);
    std::uniform_int_distribution<exp_t> n_dist(-149, 0);

    for (size_t i = 0; i < N; i++) {
        // generate a random bitstring
        uint32_t raw_bits = bits_dist(rng);
        bit_float<float> bf(raw_bits);
        float v = bf.to_float();

        // skip NaN
        if (std::isnan(v)) {
            continue;
        }

        // generate a precision [1, 24]
        prec_t prec = prec_dist(rng);

        // generate a subnormalization point [-149, 0]
        exp_t n = n_dist(rng);

        // round via both methods and compare results
        float y_expect = round(v, prec, n, rm);
        float y = experimental::round<rm>(bf, prec, n).to_float();
        EXPECT_EQ(y, y_expect);
    }
}
