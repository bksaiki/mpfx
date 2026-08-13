/// @file test_round_scaled.cpp
/// @brief Tests for the scale-and-truncate rounding implementation.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include "mpfx/round.hpp"

using mpfx::bit_float;
using mpfx::exp_t;
using mpfx::prec_t;
using mpfx::RM;
using mpfx::experimental::round_all_lost;
using mpfx::experimental::scale_bits;
using mpfx::experimental::scale_mul;

namespace {

/// @brief Renders a value as `<hex bits> (<decimal>)` for failure messages.
template <std::floating_point T>
std::string describe(T x) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setfill('0')
       << std::setw(2 * sizeof(x)) << bit_float<T>(x).to_bits()
       << std::dec << " (" << x << ")";
    return os.str();
}

/// @brief Compares two floating-point values by their encoding, so that
/// `+0` and `-0` are distinguished and NaN compares equal to itself.
template <std::floating_point T>
bool same_bits(T a, T b) {
    return bit_float<T>(a).to_bits() == bit_float<T>(b).to_bits();
}

/// @brief The name of a rounding mode, for failure messages.
const char* rm_name(RM rm) {
    switch (rm) {
    case RM::RNE: return "RNE";
    case RM::RNA: return "RNA";
    case RM::RTP: return "RTP";
    case RM::RTN: return "RTN";
    case RM::RTZ: return "RTZ";
    case RM::RAZ: return "RAZ";
    case RM::RTO: return "RTO";
    case RM::RTE: return "RTE";
    default: return "???";
    }
}

//
// scale_bits
//

/// @brief `scale_bits` must agree exactly with `ldexp` for every normal input
/// whose result is also normal.
template <std::floating_point T>
void check_scale_bits_normal() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = 200'000;

    std::mt19937_64 rng(0x5CA1E);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));

    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        if (x.ebits() == 0 || x.is_nar()) {
            continue; // subnormal, zero, or not a real
        }

        // pick `k` so that the result stays normal
        const exp_t field = static_cast<exp_t>(x.ebits() >> params_t::M);
        std::uniform_int_distribution<exp_t> k_dist(1 - field, static_cast<exp_t>(params_t::EONES) - 1 - field);
        const exp_t k = k_dist(rng);

        const T got = scale_bits(x, k).to_float();
        const T want = std::ldexp(x.to_float(), k);
        ASSERT_TRUE(same_bits(got, want))
            << "scale_bits(" << describe(x.to_float()) << ", " << k << ") = "
            << describe(got) << ", want " << describe(want);
    }
}

TEST(TestRoundScaled, ScaleBitsNormalFloat) { check_scale_bits_normal<float>(); }
TEST(TestRoundScaled, ScaleBitsNormalDouble) { check_scale_bits_normal<double>(); }

/// @brief A carry out of the largest binade must saturate onto infinity, which
/// is what the increment in the existing implementation also produces.
template <std::floating_point T>
void check_scale_bits_overflow() {
    using params_t = typename bit_float<T>::params_t;
    using limits = std::numeric_limits<T>;

    // `2^EMAX * 2 == Inf`, and the same for the negative side
    for (const bool s : {false, true}) {
        const bit_float<T> max_pow2 = bit_float<T>::make_pow2(params_t::EMAX, s);
        const T got = scale_bits(max_pow2, 1).to_float();
        const T want = s ? -limits::infinity() : limits::infinity();
        EXPECT_TRUE(same_bits(got, want))
            << "scale_bits(" << describe(max_pow2.to_float()) << ", 1) = " << describe(got);
    }
}

TEST(TestRoundScaled, ScaleBitsOverflowFloat) { check_scale_bits_overflow<float>(); }
TEST(TestRoundScaled, ScaleBitsOverflowDouble) { check_scale_bits_overflow<double>(); }

//
// scale_mul
//

/// @brief `scale_mul` must agree exactly with `ldexp` whenever the true product
/// is representable, including for subnormal inputs and subnormal results.
template <std::floating_point T>
void check_scale_mul() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = 200'000;

    std::mt19937_64 rng(0xB1A5E);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<exp_t> k_dist(2 * params_t::EMIN, 2 * params_t::EMAX);

    size_t checked = 0;
    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        if (x.is_nar() || x.is_zero()) {
            continue;
        }

        const exp_t k = k_dist(rng);
        const T v = x.to_float();
        const T want = std::ldexp(v, k);

        // only exercise `k` for which the true product is representable: the
        // round trip back through `ldexp` recovers `v` exactly in that case
        if (!std::isfinite(want) || want == static_cast<T>(0) || !same_bits(std::ldexp(want, -k), v)) {
            continue;
        }

        checked++;
        const T got = scale_mul(v, k);
        ASSERT_TRUE(same_bits(got, want))
            << "scale_mul(" << describe(v) << ", " << k << ") = "
            << describe(got) << ", want " << describe(want);
    }

    EXPECT_GT(checked, N / 100) << "too few representable cases were exercised";
}

TEST(TestRoundScaled, ScaleMulFloat) { check_scale_mul<float>(); }
TEST(TestRoundScaled, ScaleMulDouble) { check_scale_mul<double>(); }

/// @brief The cases that a single multiply cannot handle, because the scale
/// factor `2^k` is itself unrepresentable. These are exactly the scales that
/// rounding needs at the extremes, so they must be exact.
template <std::floating_point T>
void check_scale_mul_unrepresentable_factor() {
    using params_t = typename bit_float<T>::params_t;

    // scaling the smallest subnormal up, for every `k` above `EMAX`
    const T min_sub = bit_float<T>::make_pow2(params_t::EXPMIN).to_float();
    for (exp_t k = params_t::EMAX + 1; k <= -params_t::EXPMIN - 1; k++) {
        const T want = std::ldexp(min_sub, k);
        ASSERT_TRUE(same_bits(scale_mul(min_sub, k), want))
            << "scale_mul(min subnormal, " << k << ") = " << describe(scale_mul(min_sub, k))
            << ", want " << describe(want);
    }

    // and scaling back down, for every `k` below `EMIN`
    const T one = static_cast<T>(1);
    for (exp_t k = params_t::EMIN - 1; k >= params_t::EXPMIN + 1; k--) {
        const T want = std::ldexp(one, k);
        ASSERT_TRUE(same_bits(scale_mul(one, k), want))
            << "scale_mul(1, " << k << ") = " << describe(scale_mul(one, k))
            << ", want " << describe(want);
    }
}

TEST(TestRoundScaled, ScaleMulUnrepresentableFactorFloat) {
    check_scale_mul_unrepresentable_factor<float>();
}
TEST(TestRoundScaled, ScaleMulUnrepresentableFactorDouble) {
    check_scale_mul_unrepresentable_factor<double>();
}

//
// round_all_lost
//

/// @brief Dispatches `round_all_lost` on a runtime rounding mode.
template <std::floating_point T>
bit_float<T> all_lost(bit_float<T> x, exp_t n, RM rm) {
    switch (rm) {
    case RM::RNE: return round_all_lost<RM::RNE>(x, n);
    case RM::RNA: return round_all_lost<RM::RNA>(x, n);
    case RM::RTP: return round_all_lost<RM::RTP>(x, n);
    case RM::RTN: return round_all_lost<RM::RTN>(x, n);
    case RM::RTZ: return round_all_lost<RM::RTZ>(x, n);
    case RM::RAZ: return round_all_lost<RM::RAZ>(x, n);
    case RM::RTO: return round_all_lost<RM::RTO>(x, n);
    case RM::RTE: return round_all_lost<RM::RTE>(x, n);
    default: return x;
    }
}

/// @brief For split points at or above the normalized exponent, `round_all_lost`
/// must match the existing implementation in every rounding mode.
template <std::floating_point T>
void check_round_all_lost() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = 100'000;
    static constexpr RM MODES[] = {
        RM::RNE, RM::RNA, RM::RTP, RM::RTN, RM::RTZ, RM::RAZ, RM::RTO, RM::RTE,
    };

    std::mt19937_64 rng(0xA110C);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P - 1);

    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        if (x.is_nar() || x.is_zero()) {
            continue;
        }

        // pick a split point at or above `e`, staying within `EMAX` so that
        // `2^(n+1)` remains representable
        const exp_t e = x.e();
        if (e >= params_t::EMAX) {
            continue;
        }
        std::uniform_int_distribution<exp_t> n_dist(e, params_t::EMAX - 1);
        const exp_t n = n_dist(rng);
        const prec_t p = prec_dist(rng);

        for (const RM rm : MODES) {
            // `n >= e >= e - p`, so the existing implementation splits at `n` too
            const bit_float<T> want =
                mpfx::experimental::round<mpfx::Flags::NO_FLAGS>(x, p, n, rm);
            const bit_float<T> got = all_lost(x, n, rm);
            ASSERT_EQ(got.to_bits(), want.to_bits())
                << "round_all_lost<" << rm_name(rm) << ">(" << describe(x.to_float())
                << ", n=" << n << ") = " << describe(got.to_float())
                << ", want " << describe(want.to_float()) << " (p=" << p << ")";
        }
    }
}

TEST(TestRoundScaled, RoundAllLostFloat) { check_round_all_lost<float>(); }
TEST(TestRoundScaled, RoundAllLostDouble) { check_round_all_lost<double>(); }

} // namespace
