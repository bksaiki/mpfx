#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>


TEST(TestFlags, TestInvalidFlag) {
    // Test values
    const double nan = std::nan("");
    const double pos_inf = std::numeric_limits<double>::infinity();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const double zero = 0.0;
    const double pos_val = 1.5;
    const double neg_val = -2.5;

    // Simple context for testing
    const mpfx::IEEE754Context ctx(8, 32, mpfx::RM::RNE);

    // Test add: inf + (-inf) should set invalid flag
    mpfx::reset_flags();
    mpfx::add(pos_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::add(neg_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test add: NaN propagation should NOT set invalid flag
    mpfx::reset_flags();
    mpfx::add(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::add(pos_val, nan, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test add: valid operations should not set flag
    mpfx::reset_flags();
    mpfx::add(pos_inf, pos_inf, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::add(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test sub: inf - inf should set invalid flag
    mpfx::reset_flags();
    mpfx::sub(pos_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::sub(neg_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test sub: NaN propagation should NOT set invalid flag
    mpfx::reset_flags();
    mpfx::sub(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test sub: valid operations should not set flag
    mpfx::reset_flags();
    mpfx::sub(pos_inf, neg_inf, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::sub(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test mul: 0 * inf should set invalid flag
    mpfx::reset_flags();
    mpfx::mul(zero, pos_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::mul(pos_inf, zero, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::mul(zero, neg_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::mul(neg_inf, zero, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test mul: NaN propagation should NOT set invalid flag
    mpfx::reset_flags();
    mpfx::mul(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test mul: valid operations should not set flag
    mpfx::reset_flags();
    mpfx::mul(pos_inf, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::mul(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test div: 0/0 should set invalid flag
    mpfx::reset_flags();
    mpfx::div(zero, zero, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test div: inf/inf should set invalid flag
    mpfx::reset_flags();
    mpfx::div(pos_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::div(pos_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::div(neg_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::div(neg_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test div: NaN propagation should NOT set invalid flag
    mpfx::reset_flags();
    mpfx::div(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::div(pos_val, nan, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test div: valid operations should not set flag
    mpfx::reset_flags();
    mpfx::div(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::div(pos_inf, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test sqrt: negative finite number should set invalid flag
    mpfx::reset_flags();
    mpfx::sqrt(neg_val, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::sqrt(-0.5, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test sqrt: NaN propagation should NOT set invalid flag
    mpfx::reset_flags();
    mpfx::sqrt(nan, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test sqrt: valid operations should not set flag
    mpfx::reset_flags();
    mpfx::sqrt(pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::sqrt(zero, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::sqrt(pos_inf, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::sqrt(-0.0, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test fma: 0 * inf + z should set invalid flag
    mpfx::reset_flags();
    mpfx::fma(zero, pos_inf, pos_val, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::fma(pos_inf, zero, pos_val, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test fma: x * y + z where x*y = inf and z = -inf should set invalid flag
    mpfx::reset_flags();
    mpfx::fma(pos_inf, pos_val, neg_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::fma(neg_inf, pos_val, pos_inf, ctx);
    EXPECT_TRUE(mpfx::invalid_flag);

    // Test fma: NaN propagation should NOT set invalid flag
    mpfx::reset_flags();
    mpfx::fma(nan, pos_val, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::fma(pos_val, nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::fma(pos_val, pos_val, nan, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    // Test fma: valid operations should not set flag
    mpfx::reset_flags();
    mpfx::fma(pos_val, neg_val, zero, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);

    mpfx::reset_flags();
    mpfx::fma(pos_inf, pos_val, pos_inf, ctx);
    EXPECT_FALSE(mpfx::invalid_flag);
}

TEST(TestFlags, TestDivByZeroFlag) {
    // Test values
    const double nan = std::nan("");
    const double pos_inf = std::numeric_limits<double>::infinity();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const double zero = 0.0;
    const double neg_zero = -0.0;
    const double pos_val = 1.5;
    const double neg_val = -2.5;

    // Simple context for testing
    const mpfx::IEEE754Context ctx(8, 32, mpfx::RM::RNE);

    // Test div: finite non-zero / 0 should set div_by_zero flag
    mpfx::reset_flags();
    mpfx::div(pos_val, zero, ctx);
    EXPECT_TRUE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(neg_val, zero, ctx);
    EXPECT_TRUE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(pos_val, neg_zero, ctx);
    EXPECT_TRUE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(neg_val, neg_zero, ctx);
    EXPECT_TRUE(mpfx::div_by_zero_flag);

    // Test div: 0/0 should NOT set div_by_zero flag (it sets invalid instead)
    mpfx::reset_flags();
    mpfx::div(zero, zero, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(neg_zero, zero, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    // Test div: inf/0 should NOT set div_by_zero flag (inf/0 = inf, not a special case)
    mpfx::reset_flags();
    mpfx::div(pos_inf, zero, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(neg_inf, zero, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    // Test div: NaN/0 should NOT set div_by_zero flag
    mpfx::reset_flags();
    mpfx::div(nan, zero, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    // Test div: x/y where y != 0 should NOT set div_by_zero flag
    mpfx::reset_flags();
    mpfx::div(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(zero, pos_val, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(pos_inf, pos_val, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);

    mpfx::reset_flags();
    mpfx::div(pos_val, pos_inf, ctx);
    EXPECT_FALSE(mpfx::div_by_zero_flag);
}

TEST(TestFlags, TestOverflowFlag) {
    static constexpr size_t N = 1'000'000;
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
    static constexpr size_t N = 1'000'000;
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
    static constexpr size_t N = 1'000'000;
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
        const auto y_unbound = mpfx::round(x, p, std::nullopt, rm);
        mpfx::reset_flags();

        const auto y = mpfx::round(x, p, n, rm);
        (void) y;

        // compute exponents
        const mpfx::exp_t ye = (y_unbound == 0.0) ? 0 : std::ilogb(y_unbound);
        const mpfx::exp_t emin = n + static_cast<mpfx::exp_t>(p);

        // check inexact flag
        EXPECT_EQ(mpfx::tiny_after_rounding_flag, y_unbound == 0 || ye < emin);

        // reset flag
        mpfx::reset_flags();
    }
}

TEST(TestFlags, TestInexactFlag) {
    static constexpr size_t N = 1'000'000;
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
    static constexpr size_t N = 1'000'000;
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
        (void) y;

        // check underflow flag
        EXPECT_EQ(mpfx::underflow_before_rounding_flag, mpfx::inexact_flag && mpfx::tiny_before_rounding_flag);

        // reset flag
        mpfx::reset_flags();
    }
}

TEST(TestFlags, TestUnderflowAfterFlag) {
    static constexpr size_t N = 1'000'000;
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
        (void) y;

        // check underflow flag
        EXPECT_EQ(mpfx::underflow_after_rounding_flag, mpfx::inexact_flag && mpfx::tiny_after_rounding_flag);

        // reset flag
        mpfx::reset_flags();
    }
}

TEST(TestFlags, TestCarry) {
    static constexpr size_t N = 1'000'000;
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

TEST(TestFlags, TestTinyExamples) {
    constexpr mpfx::prec_t PREC = 2;
    constexpr mpfx::exp_t N = -2;
    constexpr mpfx::RM RM = mpfx::RM::RNE;

    const double x0 = 1.0;
    mpfx::round(x0, PREC, N, RM);
    EXPECT_FALSE(mpfx::tiny_before_rounding_flag);
    EXPECT_FALSE(mpfx::tiny_after_rounding_flag);
    mpfx::reset_flags();

    const double x1 = 0.9375;
    mpfx::round(x1, PREC, N, RM);
    EXPECT_TRUE(mpfx::tiny_before_rounding_flag);
    EXPECT_FALSE(mpfx::tiny_after_rounding_flag);
    mpfx::reset_flags();

    const double x2 = 0.875;
    mpfx::round(x2, PREC, N, RM);
    EXPECT_TRUE(mpfx::tiny_before_rounding_flag);
    EXPECT_FALSE(mpfx::tiny_after_rounding_flag);
    mpfx::reset_flags();

    const double x3 = 0.8125;
    mpfx::round(x3, PREC, N, RM);
    EXPECT_TRUE(mpfx::tiny_before_rounding_flag);
    EXPECT_TRUE(mpfx::tiny_after_rounding_flag);
    mpfx::reset_flags();

    const double x4 = 0.75;
    mpfx::round(x4, PREC, N, RM);
    EXPECT_TRUE(mpfx::tiny_before_rounding_flag);
    EXPECT_TRUE(mpfx::tiny_after_rounding_flag);
    mpfx::reset_flags();
}
