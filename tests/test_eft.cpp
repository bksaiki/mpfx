#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>
#include <mpfx/engine_eft.hpp>
#include <mpfx.hpp>

inline bool nonoverlapping_check(double x, double y) {
    if (x == 0.0) {
        return y == 0.0;
    } else if (y == 0.0) {
        return true;
    } else {
        // unpack floating-point values
        const auto xparts = mpfx::unpack_float(x);
        const auto yparts = mpfx::unpack_float(y);
        const auto ex = std::get<1>(xparts);
        const auto ey = std::get<1>(yparts);
        return ex - ey >= 53;
    }
}

TEST(TestEFT, TestEFTAdd3) {
    static constexpr size_t N = 1'000'000;

    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> exp_dist(-150, 150);

    for (size_t i = 0; i < N; i++) {
        const double x0 = std::ldexp(dist(rng), exp_dist(rng));
        const double x1 = std::ldexp(dist(rng), exp_dist(rng));
        const double x2 = std::ldexp(dist(rng), exp_dist(rng));

        // std::cout << "Test case " << i << ": " << x0 << ", " << x1 << ", " << x2 << ", " << x3 << std::endl;
        const auto [s0, s1, s2] = mpfx::engine_eft::eft_add3(x0, x1, x2);
        EXPECT_TRUE(nonoverlapping_check(s0, s1));
        EXPECT_TRUE(nonoverlapping_check(s1, s2));
    }

}


TEST(TestEFT, TestEFTAdd4) {
    static constexpr size_t N = 1'000'000;

    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> exp_dist(-150, 150);

    for (size_t i = 0; i < N; i++) {
        const double x0 = std::ldexp(dist(rng), exp_dist(rng));
        const double x1 = std::ldexp(dist(rng), exp_dist(rng));
        const double x2 = std::ldexp(dist(rng), exp_dist(rng));
        const double x3 = std::ldexp(dist(rng), exp_dist(rng));

        // std::cout << "Test case " << i << ": " << x0 << ", " << x1 << ", " << x2 << ", " << x3 << std::endl;
        const auto [s0, s1, s2, s3] = mpfx::engine_eft::eft_add4(x0, x1, x2, x3);
        EXPECT_TRUE(nonoverlapping_check(s0, s1));
        EXPECT_TRUE(nonoverlapping_check(s1, s2));
        EXPECT_TRUE(nonoverlapping_check(s2, s3));
    }

}

// Independent round-to-odd reference for `round_finalize`. Given the EFT
// invariant high = RN(v) and low = v - high (exact), it returns the round-to-odd
// of v at the container precision: of the two representable values bracketing v,
// the one with an odd mantissa LSB. This uses `nextafter` + bit inspection, a
// different mechanism than `round_finalize`'s magnitude arithmetic, so the two
// agreeing is meaningful.
template <std::floating_point T>
static T rto_reference(T high, T low) {
    using U = typename mpfx::float_params<T>::uint_t;
    if (low == static_cast<T>(0)) {
        return high; // exact: v == high
    }
    // v = high + low lies strictly between `high` and its neighbor in the
    // direction of `low`'s sign.
    const T dir = low > static_cast<T>(0)
        ? std::numeric_limits<T>::infinity()
        : -std::numeric_limits<T>::infinity();
    const T neighbor = std::nextafter(high, dir);
    // exactly one of {high, neighbor} has an odd mantissa LSB -> that is the RTO
    const bool high_odd = (std::bit_cast<U>(high) & static_cast<U>(1)) != 0;
    return high_odd ? high : neighbor;
}

// Property test: for valid EFT pairs (from two_sum over both signs and a wide
// exponent range), round_finalize must equal the independent RTO reference.
template <std::floating_point T>
static void check_round_finalize_matches_rto() {
    std::mt19937_64 rng(0xC0FFEEu);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> exp_dist(-60, 60);

    for (size_t i = 0; i < 2'000'000; i++) {
        const T x = static_cast<T>(std::ldexp(dist(rng), exp_dist(rng)));
        const T y = static_cast<T>(std::ldexp(dist(rng), exp_dist(rng)));

        // two_sum yields the exact (high, low) pair round_finalize consumes
        const auto [high, low] = mpfx::engine_eft::two_sum(x, y);
        if (high == static_cast<T>(0) || !std::isfinite(high) || !std::isfinite(low)) {
            continue; // round_finalize assumes a finite, non-zero `high`
        }

        const T got = mpfx::engine_eft::round_finalize(high, low);
        const T want = rto_reference(high, low);
        ASSERT_EQ(got, want)
            << "high=" << high << " low=" << low << " (x=" << x << ", y=" << y << ")";
    }
}

TEST(TestEFT, RoundFinalizeMatchesRTOFloat) {
    check_round_finalize_matches_rto<float>();
}

TEST(TestEFT, RoundFinalizeMatchesRTODouble) {
    check_round_finalize_matches_rto<double>();
}

TEST(TestEFT, RoundFinalizeRegression) {
    // Regression for the negative-`high`, opposite-sign-`low` case. The exact
    // value lies one (sub-ULP) step toward zero from `high`, so the RTO result
    // must move toward zero (smaller magnitude) and be odd. The original code
    // moved away from zero, producing 0xBF8E8001 instead of 0xBF8E7FFF.
    const float high = std::bit_cast<float>(0xBF8E8000u); // -1.11328125 (even LSB)
    const float low = 5.0e-8f;                            // positive, < half ULP (~5.96e-8)
    EXPECT_EQ(std::bit_cast<uint32_t>(mpfx::engine_eft::round_finalize(high, low)),
              0xBF8E7FFFu);

    // Mirror case: positive `high`, opposite-sign (negative) `low` -> toward zero.
    const float phigh = std::bit_cast<float>(0x3F8E8000u); // +1.11328125
    EXPECT_EQ(std::bit_cast<uint32_t>(mpfx::engine_eft::round_finalize(phigh, -5.0e-8f)),
              0x3F8E7FFFu);

    // Same-sign `low` keeps `high`'s binade and just jams the sticky bit (odd),
    // moving away from zero when `high` is even.
    EXPECT_EQ(std::bit_cast<uint32_t>(mpfx::engine_eft::round_finalize(high, -5.0e-8f)),
              0xBF8E8001u); // negative high, negative low -> away from zero
    EXPECT_EQ(std::bit_cast<uint32_t>(mpfx::engine_eft::round_finalize(phigh, 5.0e-8f)),
              0x3F8E8001u); // positive high, positive low -> away from zero

    // Exact result (low == 0) is returned unchanged.
    EXPECT_EQ(std::bit_cast<uint32_t>(mpfx::engine_eft::round_finalize(high, 0.0f)),
              0xBF8E8000u);
}
