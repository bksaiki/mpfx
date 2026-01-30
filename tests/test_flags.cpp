#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>


TEST(TestFlags, TestInvalidFlag) {

}

TEST(TestFlags, TestDivByZeroFlag) {

}

TEST(TestFlags, TestOverflowFlag) {
    static constexpr size_t N = 100'000;
    static constexpr mpfx::prec_t MAX_PREC = 8;
    static constexpr mpfx::exp_t MAX_EXP = 4;
    static constexpr mpfx::exp_t MIN_EXP = -4;
    std::vector<mpfx::RM> RMS = {
        mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP,
        mpfx::RM::RTN, mpfx::RM::RTZ, mpfx::RM::RAZ,
        mpfx::RM::RTO, mpfx::RM::RTE
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // distributions
    std::uniform_int_distribution<int> s_dist(0, 1);
    std::uniform_int_distribution<mpfx::mant_t> c_dist(0, (1 << MAX_PREC) - 1);
    std::uniform_int_distribution<mpfx::exp_t> exp_dist(MIN_EXP, MAX_EXP);
    std::uniform_int_distribution<mpfx::prec_t> prec_dist(1, MAX_PREC);
    std::uniform_int_distribution<size_t> rm_dist(0, RMS.size() - 1);

    for (size_t i = 0; i < N; i++) {
        // rounding parameters
        const auto p = prec_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // significand sampler
        std::uniform_int_distribution<mpfx::mant_t> c_dist(0, (1 << p) - 1);

        // generate random number
        const bool s1 = s_dist(rng) != 0;
        const auto c1 = c_dist(rng);
        const auto exp1 = exp_dist(rng);
        const double x = mpfx::make_double(s1, exp1, c1);

        // generate random overflow threshold
        const auto c2 = c_dist(rng);
        const auto exp2 = exp_dist(rng);
        const auto bound = mpfx::make_double(false, exp2, c2);

        // rounding context
        const mpfx::MPBContext ctx(p, MIN_EXP, rm, bound);

        // round with context
        const auto y = ctx.round(x);
        (void) y;

        // check overflow flag
        EXPECT_EQ(mpfx::overflow_flag, std::abs(x) > bound);
        if (mpfx::overflow_flag) {
            EXPECT_TRUE(mpfx::inexact_flag);
        }

        // reset flag
        mpfx::reset_flags();
    }

}

TEST(TestFlags, TestTinyBeforeFlag) {
    static constexpr size_t N = 100'000;
    static constexpr mpfx::prec_t MAX_PREC = 8;
    static constexpr mpfx::exp_t MAX_EXP = 4;
    static constexpr mpfx::exp_t MIN_EXP = -4;
    std::vector<mpfx::RM> RMS = {
        mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP,
        mpfx::RM::RTN, mpfx::RM::RTZ, mpfx::RM::RAZ,
        mpfx::RM::RTO, mpfx::RM::RTE
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // distributions
    std::uniform_int_distribution<int> s_dist(0, 1);
    std::uniform_int_distribution<mpfx::mant_t> c_dist(0, (1 << MAX_PREC) - 1);
    std::uniform_int_distribution<mpfx::exp_t> exp_dist(MIN_EXP, MAX_EXP);
    std::uniform_int_distribution<mpfx::prec_t> prec_dist(1, MAX_PREC);
    std::uniform_int_distribution<mpfx::exp_t> n_dist(MIN_EXP - 1, MAX_EXP);
    std::uniform_int_distribution<size_t> rm_dist(0, RMS.size() - 1);

    for (size_t i = 0; i < N; i++) {
        // generate random number
        const bool s = s_dist(rng) != 0;
        const auto c = c_dist(rng);
        const auto exp = exp_dist(rng);
        const double x = mpfx::make_double(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // compute exponents
        const mpfx::exp_t xe = (x == 0.0) ? 0 : std::ilogb(x);
        const mpfx::exp_t emin = n + static_cast<mpfx::exp_t>(p);

        // round
        const auto y = mpfx::round(x, p, n, rm);
        (void) y;

        // check inexact flag
        EXPECT_EQ(mpfx::tiny_before_rounding_flag, x == 0 || xe < emin);

        // reset flag
        mpfx::reset_flags();
    }
}

TEST(TestFlags, TestTinyAfterFlag) {

}

TEST(TestFlags, TestInexactFlag) {
    static constexpr size_t N = 100'000;
    static constexpr mpfx::prec_t MAX_PREC = 8;
    static constexpr mpfx::exp_t MAX_EXP = 4;
    static constexpr mpfx::exp_t MIN_EXP = -4;
    std::vector<mpfx::RM> RMS = {
        mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP,
        mpfx::RM::RTN, mpfx::RM::RTZ, mpfx::RM::RAZ,
        mpfx::RM::RTO, mpfx::RM::RTE
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // distributions
    std::uniform_int_distribution<int> s_dist(0, 1);
    std::uniform_int_distribution<mpfx::mant_t> c_dist(0, (1 << MAX_PREC) - 1);
    std::uniform_int_distribution<mpfx::exp_t> exp_dist(MIN_EXP, MAX_EXP);
    std::uniform_int_distribution<mpfx::prec_t> prec_dist(1, MAX_PREC);
    std::uniform_int_distribution<mpfx::exp_t> n_dist(MIN_EXP - 1, MAX_EXP);
    std::uniform_int_distribution<size_t> rm_dist(0, RMS.size() - 1);

    for (size_t i = 0; i < N; i++) {
        // generate random number
        const bool s = s_dist(rng) != 0;
        const auto c = c_dist(rng);
        const auto exp = exp_dist(rng);
        const double x = mpfx::make_double(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y = mpfx::round(x, p, n, rm);

        // check inexact flag
        EXPECT_EQ(mpfx::inexact_flag, (x != y));

        // reset flag
        mpfx::reset_flags();
    }
}

TEST(TestFlags, TestUnderflowBeforeFlag) {

}

TEST(TestFlags, TestUnderflowAfterFlag) {

}

TEST(TestFlags, TestCarry) {
    static constexpr size_t N = 100'000;
    static constexpr mpfx::prec_t MAX_PREC = 8;
    static constexpr mpfx::exp_t MAX_EXP = 4;
    static constexpr mpfx::exp_t MIN_EXP = -4;
    std::vector<mpfx::RM> RMS = {
        mpfx::RM::RNE, mpfx::RM::RNA, mpfx::RM::RTP,
        mpfx::RM::RTN, mpfx::RM::RTZ, mpfx::RM::RAZ,
        mpfx::RM::RTO, mpfx::RM::RTE
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // distributions
    std::uniform_int_distribution<int> s_dist(0, 1);
    std::uniform_int_distribution<mpfx::mant_t> c_dist(0, (1 << MAX_PREC) - 1);
    std::uniform_int_distribution<mpfx::exp_t> exp_dist(MIN_EXP, MAX_EXP);
    std::uniform_int_distribution<mpfx::prec_t> prec_dist(1, MAX_PREC);
    std::uniform_int_distribution<mpfx::exp_t> n_dist(MIN_EXP - 1, MAX_EXP);
    std::uniform_int_distribution<size_t> rm_dist(0, RMS.size() - 1);

    for (size_t i = 0; i < N; i++) {
        // generate random number
        const bool s = s_dist(rng) != 0;
        const auto c = c_dist(rng);
        const auto exp = exp_dist(rng);
        const double x = mpfx::make_double(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y = mpfx::round(x, p, n, rm);

        // compute exponents
        const auto xe = (x == 0.0) ? 0 : std::ilogb(x);
        const auto ye = (y == 0.0) ? 0 : std::ilogb(y);

        // check for carry
        EXPECT_EQ(mpfx::carry_flag, x != 0 && y != 0 && ye > xe);

        // reset flags
        mpfx::reset_flags();
    }
}
