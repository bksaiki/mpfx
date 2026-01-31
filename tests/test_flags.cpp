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
    mpfx::flags.reset();
    mpfx::add(pos_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::add(neg_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test add: NaN propagation should NOT set invalid flag
    mpfx::flags.reset();
    mpfx::add(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::add(pos_val, nan, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test add: valid operations should not set flag
    mpfx::flags.reset();
    mpfx::add(pos_inf, pos_inf, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::add(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test sub: inf - inf should set invalid flag
    mpfx::flags.reset();
    mpfx::sub(pos_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::sub(neg_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test sub: NaN propagation should NOT set invalid flag
    mpfx::flags.reset();
    mpfx::sub(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test sub: valid operations should not set flag
    mpfx::flags.reset();
    mpfx::sub(pos_inf, neg_inf, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::sub(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test mul: 0 * inf should set invalid flag
    mpfx::flags.reset();
    mpfx::mul(zero, pos_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::mul(pos_inf, zero, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::mul(zero, neg_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::mul(neg_inf, zero, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test mul: NaN propagation should NOT set invalid flag
    mpfx::flags.reset();
    mpfx::mul(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test mul: valid operations should not set flag
    mpfx::flags.reset();
    mpfx::mul(pos_inf, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::mul(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test div: 0/0 should set invalid flag
    mpfx::flags.reset();
    mpfx::div(zero, zero, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test div: inf/inf should set invalid flag
    mpfx::flags.reset();
    mpfx::div(pos_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::div(pos_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::div(neg_inf, pos_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::div(neg_inf, neg_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test div: NaN propagation should NOT set invalid flag
    mpfx::flags.reset();
    mpfx::div(nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::div(pos_val, nan, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test div: valid operations should not set flag
    mpfx::flags.reset();
    mpfx::div(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::div(pos_inf, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test sqrt: negative finite number should set invalid flag
    mpfx::flags.reset();
    mpfx::sqrt(neg_val, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::sqrt(-0.5, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test sqrt: NaN propagation should NOT set invalid flag
    mpfx::flags.reset();
    mpfx::sqrt(nan, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test sqrt: valid operations should not set flag
    mpfx::flags.reset();
    mpfx::sqrt(pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::sqrt(zero, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::sqrt(pos_inf, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::sqrt(-0.0, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test fma: 0 * inf + z should set invalid flag
    mpfx::flags.reset();
    mpfx::fma(zero, pos_inf, pos_val, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::fma(pos_inf, zero, pos_val, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test fma: x * y + z where x*y = inf and z = -inf should set invalid flag
    mpfx::flags.reset();
    mpfx::fma(pos_inf, pos_val, neg_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::fma(neg_inf, pos_val, pos_inf, ctx);
    EXPECT_TRUE(mpfx::flags.invalid());

    // Test fma: NaN propagation should NOT set invalid flag
    mpfx::flags.reset();
    mpfx::fma(nan, pos_val, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::fma(pos_val, nan, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::fma(pos_val, pos_val, nan, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    // Test fma: valid operations should not set flag
    mpfx::flags.reset();
    mpfx::fma(pos_val, neg_val, zero, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());

    mpfx::flags.reset();
    mpfx::fma(pos_inf, pos_val, pos_inf, ctx);
    EXPECT_FALSE(mpfx::flags.invalid());
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
    mpfx::flags.reset();
    mpfx::div(pos_val, zero, ctx);
    EXPECT_TRUE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(neg_val, zero, ctx);
    EXPECT_TRUE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(pos_val, neg_zero, ctx);
    EXPECT_TRUE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(neg_val, neg_zero, ctx);
    EXPECT_TRUE(mpfx::flags.div_by_zero());

    // Test div: 0/0 should NOT set div_by_zero flag (it sets invalid instead)
    mpfx::flags.reset();
    mpfx::div(zero, zero, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(neg_zero, zero, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    // Test div: inf/0 should NOT set div_by_zero flag (inf/0 = inf, not a special case)
    mpfx::flags.reset();
    mpfx::div(pos_inf, zero, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(neg_inf, zero, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    // Test div: NaN/0 should NOT set div_by_zero flag
    mpfx::flags.reset();
    mpfx::div(nan, zero, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    // Test div: x/y where y != 0 should NOT set div_by_zero flag
    mpfx::flags.reset();
    mpfx::div(pos_val, neg_val, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(zero, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(pos_inf, pos_val, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());

    mpfx::flags.reset();
    mpfx::div(pos_val, pos_inf, ctx);
    EXPECT_FALSE(mpfx::flags.div_by_zero());
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
        const double x = mpfx::make_float<double>(s1, exp1, c1);

        // generate random overflow threshold
        const auto c2 = c_dist(rng);
        const auto exp2 = exp_dist(rng);
        const auto bound = mpfx::make_float<double>(false, exp2, c2);

        // rounding context
        const mpfx::MPBContext ctx(p, MIN_EXP, rm, bound);

        // round with context
        const auto y = ctx.round(x);
        (void) y;

        // check overflow flag
        EXPECT_EQ(mpfx::flags.overflow(), std::abs(x) > bound);
        if (mpfx::flags.overflow()) {
            EXPECT_TRUE(mpfx::flags.inexact());
        }

        // reset flag
        mpfx::flags.reset();
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
        const double x = mpfx::make_float<double>(s, exp, c);

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
        EXPECT_EQ(mpfx::flags.tiny_before_rounding(), x == 0 || xe < emin);

        // reset flag
        mpfx::flags.reset();
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
        const double x = mpfx::make_float<double>(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y_unbound = mpfx::round(x, p, std::nullopt, rm);
        mpfx::flags.reset();

        const auto y = mpfx::round(x, p, n, rm);
        (void) y;

        // compute exponents
        const mpfx::exp_t ye = (y_unbound == 0.0) ? 0 : std::ilogb(y_unbound);
        const mpfx::exp_t emin = n + static_cast<mpfx::exp_t>(p);

        // check inexact flag
        EXPECT_EQ(mpfx::flags.tiny_after_rounding(), y_unbound == 0 || ye < emin);

        // reset flag
        mpfx::flags.reset();
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
        const double x = mpfx::make_float<double>(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y = mpfx::round(x, p, n, rm);

        // check inexact flag
        EXPECT_EQ(mpfx::flags.inexact(), (x != y));

        // reset flag
        mpfx::flags.reset();
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
        const double x = mpfx::make_float<double>(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y = mpfx::round(x, p, n, rm);
        (void) y;

        // check underflow flag
        EXPECT_EQ(mpfx::flags.underflow_before_rounding(), mpfx::flags.inexact() && mpfx::flags.tiny_before_rounding());

        // reset flag
        mpfx::flags.reset();
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
        const double x = mpfx::make_float<double>(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y = mpfx::round(x, p, n, rm);
        (void) y;

        // check underflow flag
        EXPECT_EQ(mpfx::flags.underflow_after_rounding(), mpfx::flags.inexact() && mpfx::flags.tiny_after_rounding());

        // reset flag
        mpfx::flags.reset();
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
        const double x = mpfx::make_float<double>(s, exp, c);

        // generate random precision and n
        const auto p = prec_dist(rng);
        const auto n = n_dist(rng);
        const auto rm = RMS[rm_dist(rng)];

        // round
        const auto y = mpfx::round(x, p, n, rm);

        // compute exponents
        const mpfx::exp_t emin = n + static_cast<mpfx::exp_t>(p);
        const auto xe = (x == 0.0) ? 0 : std::ilogb(x);
        const auto ye = (y == 0.0) ? 0 : std::ilogb(y);

        // check for carry
        EXPECT_EQ(mpfx::flags.carry(), x != 0 && y != 0 && ye > xe && xe >= emin);

        // reset flags
        mpfx::flags.reset();
    }
}

TEST(TestFlags, TestTinyExamples) {
    constexpr mpfx::prec_t PREC = 2;
    constexpr mpfx::exp_t N = -2;
    constexpr mpfx::RM RM = mpfx::RM::RNE;

    const double x0 = 1.0;
    mpfx::round(x0, PREC, N, RM);
    EXPECT_FALSE(mpfx::flags.tiny_before_rounding());
    EXPECT_FALSE(mpfx::flags.tiny_after_rounding());
    mpfx::flags.reset();

    const double x1 = 0.9375;
    mpfx::round(x1, PREC, N, RM);
    EXPECT_TRUE(mpfx::flags.tiny_before_rounding());
    EXPECT_FALSE(mpfx::flags.tiny_after_rounding());
    mpfx::flags.reset();

    const double x2 = 0.875;
    mpfx::round(x2, PREC, N, RM);
    EXPECT_TRUE(mpfx::flags.tiny_before_rounding());
    EXPECT_FALSE(mpfx::flags.tiny_after_rounding());
    mpfx::flags.reset();

    const double x3 = 0.8125;
    mpfx::round(x3, PREC, N, RM);
    EXPECT_TRUE(mpfx::flags.tiny_before_rounding());
    EXPECT_TRUE(mpfx::flags.tiny_after_rounding());
    mpfx::flags.reset();

    const double x4 = 0.75;
    mpfx::round(x4, PREC, N, RM);
    EXPECT_TRUE(mpfx::flags.tiny_before_rounding());
    EXPECT_TRUE(mpfx::flags.tiny_after_rounding());
    mpfx::flags.reset();
}
