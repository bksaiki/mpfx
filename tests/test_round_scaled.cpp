/// @file test_round_scaled.cpp
/// @brief Tests for the scale-and-truncate rounding implementation.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <random>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

//
// round_scaled
//

constexpr RM MODES[] = {
    RM::RNE, RM::RNA, RM::RTP, RM::RTN, RM::RTZ, RM::RAZ, RM::RTO, RM::RTE,
};

/// @brief Renders an optional subnormalization point for failure messages.
std::string describe_n(std::optional<exp_t> n) {
    return n.has_value() ? std::to_string(*n) : std::string("none");
}

/// @brief `round_scaled` must agree with `round` bit for bit, in every rounding
/// mode, for one `(x, p, n)` triple.
template <std::floating_point T, bool FieldScale, bool FpReduce>
void expect_matches(bit_float<T> x, prec_t p, std::optional<exp_t> n, const std::string& label) {
    for (const RM rm : MODES) {
        const bit_float<T> want = mpfx::experimental::round<mpfx::Flags::NO_FLAGS>(x, p, n, rm);
        bit_float<T> got;
        if constexpr (FpReduce) {
            got = mpfx::experimental::round_scaled_fp<mpfx::Flags::NO_FLAGS, FieldScale>(x, p, n, rm);
        } else {
            got = mpfx::experimental::round_scaled<mpfx::Flags::NO_FLAGS, FieldScale>(x, p, n, rm);
        }
        ASSERT_EQ(got.to_bits(), want.to_bits())
            << "round_scaled" << (FpReduce ? "_fp" : "") << "<" << rm_name(rm) << ">("
            << describe(x.to_float())
            << ", p=" << p << ", n=" << describe_n(n) << ") = " << describe(got.to_float())
            << ", want " << describe(want.to_float())
            << " [" << label << ", field_scale=" << FieldScale << "]";
    }
}

/// @brief Interesting values in the container format.
template <std::floating_point T>
std::vector<std::pair<std::string, bit_float<T>>> edge_values() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    const auto raw = [](uint64_t b) { return bit_float<T>(static_cast<uint_t>(b)); };
    const uint64_t max_finite = ((params_t::EONES - 1) << params_t::M) | params_t::MMASK;

    return {
        {"+0", raw(0)},
        {"-0", raw(params_t::SMASK)},
        {"+inf", raw(params_t::EMASK)},
        {"-inf", raw(params_t::SMASK | params_t::EMASK)},
        {"nan", raw(params_t::EMASK | 1)},
        {"min subnormal", raw(1)},
        {"-min subnormal", raw(params_t::SMASK | 1)},
        {"max subnormal", raw(params_t::MMASK)},
        {"min normal", bit_float<T>::make_pow2(params_t::EMIN)},
        {"-min normal", bit_float<T>::make_pow2(params_t::EMIN, true)},
        {"just above min normal", raw(params_t::IMPLICIT1 | 1)},
        {"max finite", raw(max_finite)},
        {"-max finite", raw(params_t::SMASK | max_finite)},
        {"2^EMAX", bit_float<T>::make_pow2(params_t::EMAX)},
        {"one", bit_float<T>(static_cast<T>(1))},
        {"-one", bit_float<T>(static_cast<T>(-1))},
        {"all mantissa bits", raw(params_t::IMPLICIT1 | params_t::MMASK)},
    };
}

/// @brief Every interesting value crossed with every interesting `(p, n)`.
template <std::floating_point T, bool FieldScale, bool FpReduce>
void check_round_scaled_edges() {
    using params_t = typename bit_float<T>::params_t;
    static constexpr prec_t P = params_t::P;

    const std::vector<prec_t> precs = {1, 2, P / 2, P - 1};
    const std::vector<std::optional<exp_t>> ns = {
        std::nullopt,
        params_t::EXPMIN - 1,           // the lowest legal subnormalization point
        params_t::EXPMIN + 1,
        params_t::EMIN - static_cast<exp_t>(P), // the usual IEEE 754 choice
        -1,
        params_t::EMAX - 1,             // forces every digit to be discarded
    };

    for (const auto& [name, x] : edge_values<T>()) {
        for (const prec_t p : precs) {
            for (const auto& n : ns) {
                expect_matches<T, FieldScale, FpReduce>(x, p, n, name);
            }
        }
    }
}


/// @brief Values sitting exactly at, just above, and just below the halfway
/// point for each target precision - the cases where the tie-breaking rules and
/// the parity of the last kept digit actually matter.
template <std::floating_point T, bool FieldScale, bool FpReduce>
void check_round_scaled_halfway() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr prec_t M = params_t::M;

    // `1 + 2^-p` splits at `n = -p` with a lost part of exactly `2^-p`
    for (prec_t p = 1; p < params_t::P; p++) {
        const uint_t one = params_t::IMPLICIT1;
        const auto at = static_cast<uint_t>(one | (static_cast<uint_t>(1) << (M - p)));
        expect_matches<T, FieldScale, FpReduce>(bit_float<T>(at), p, std::nullopt, "halfway, even");
        expect_matches<T, FieldScale, FpReduce>(
            bit_float<T>(static_cast<uint_t>(params_t::SMASK | at)), p, std::nullopt, "-halfway, even");

        // set the last kept digit to make it odd
        if (p >= 2) {
            const auto odd = static_cast<uint_t>(at | (static_cast<uint_t>(1) << (M - p + 1)));
            expect_matches<T, FieldScale, FpReduce>(bit_float<T>(odd), p, std::nullopt, "halfway, odd");
        }

        // one digit below the halfway digit puts the value off the tie
        if (p + 1 <= M) {
            const auto above = static_cast<uint_t>(at | (static_cast<uint_t>(1) << (M - p - 1)));
            const auto below = static_cast<uint_t>(one | (static_cast<uint_t>(1) << (M - p - 1)));
            expect_matches<T, FieldScale, FpReduce>(bit_float<T>(above), p, std::nullopt, "above halfway");
            expect_matches<T, FieldScale, FpReduce>(bit_float<T>(below), p, std::nullopt, "below halfway");
        }
    }
}


/// @brief Subnormal inputs shallow enough to keep a digit, which is where the
/// multiplying scale must split its factor in two.
template <std::floating_point T, bool FieldScale, bool FpReduce>
void check_round_scaled_deep_subnormal() {
    using params_t = typename bit_float<T>::params_t;

    // a subnormal with a few digits, low enough that `2^-(n+1)` overflows
    for (exp_t e = params_t::EXPMIN + 1; e <= params_t::EXPMIN + 4; e++) {
        const T v = bit_float<T>::make_pow2(e).to_float()
                  + bit_float<T>::make_pow2(params_t::EXPMIN).to_float();
        for (prec_t p = 1; p < params_t::P; p++) {
            expect_matches<T, FieldScale, FpReduce>(bit_float<T>(v), p, std::nullopt, "deep subnormal");
            expect_matches<T, FieldScale, FpReduce>(bit_float<T>(-v), p, std::nullopt, "-deep subnormal");
            expect_matches<T, FieldScale, FpReduce>(bit_float<T>(v), p, params_t::EXPMIN - 1, "deep subnormal");
        }
    }
}


/// @brief Randomized differential test against the existing implementation.
template <std::floating_point T, bool FieldScale, bool FpReduce>
void check_round_scaled_random() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = 250'000;

    std::mt19937_64 rng(0x5EEDU);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P - 1);
    std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMAX - 1);
    std::bernoulli_distribution has_n_dist(0.5);

    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        const prec_t p = prec_dist(rng);
        const std::optional<exp_t> n =
            has_n_dist(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;
        expect_matches<T, FieldScale, FpReduce>(x, p, n, "random");
    }
}

/// @brief Randomized differential test restricted to subnormal inputs.
///
/// Uniform bit patterns are subnormal only about once in five thousand draws for
/// `float` and once in two thousand for `double`, which leaves the multiplying
/// scale and the `n_act < EXPMIN` fast path barely exercised by
/// `check_round_scaled_random`. Drawing subnormals directly fixes that.
template <std::floating_point T, bool FieldScale, bool FpReduce>
void check_round_scaled_random_subnormal() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = 100'000;

    std::mt19937_64 rng(0x50BU);
    std::uniform_int_distribution<uint_t> mant_dist(1, static_cast<uint_t>(params_t::MMASK));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P - 1);
    // keep `n` near the bottom of the format, where subnormals actually split
    std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMIN);
    std::bernoulli_distribution coin(0.5);

    for (size_t i = 0; i < N; i++) {
        uint_t bits = mant_dist(rng);
        if (coin(rng)) {
            bits |= static_cast<uint_t>(params_t::SMASK);
        }
        const bit_float<T> x(bits);
        const prec_t p = prec_dist(rng);
        const std::optional<exp_t> n =
            coin(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;
        expect_matches<T, FieldScale, FpReduce>(x, p, n, "random subnormal");
    }
}


/// @brief Confirms the randomized test actually reaches every path in
/// `round_scaled`, so that coverage cannot quietly disappear if the input
/// distributions are ever retuned. The counts must stay in step with
/// `check_round_scaled_random`.
template <std::floating_point T>
void check_round_scaled_coverage() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = 250'000;

    std::mt19937_64 rng(0x5EEDU);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P - 1);
    std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMAX - 1);
    std::bernoulli_distribution has_n_dist(0.5);

    size_t nar_or_zero = 0, below_split = 0, all_lost = 0, field_scale = 0, mul_scale = 0;
    size_t exact = 0, halfway = 0, inexact = 0;

    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        const prec_t p = prec_dist(rng);
        const std::optional<exp_t> n =
            has_n_dist(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;

        if (x.is_nar() || x.is_zero()) {
            nar_or_zero++;
            continue;
        }

        const exp_t e = x.e();
        const exp_t n_min = e - static_cast<exp_t>(p);
        const exp_t n_act = n.has_value() ? std::max(n_min, *n) : n_min;

        if (n_act < params_t::EXPMIN) {
            below_split++;
            continue;
        }
        if (n_act >= e) {
            all_lost++;
            continue;
        }

        if (x.ebits() != 0) {
            field_scale++;
        } else {
            mul_scale++;
        }

        // classify the lost digits using the existing split
        const auto [hi, rs] = x.split_rs(n_act);
        (void) hi;
        if (rs == mpfx::RoundRS::EXACT) {
            exact++;
        } else if (rs == mpfx::RoundRS::EXACT_HALFWAY) {
            halfway++;
        } else {
            inexact++;
        }
    }

    std::cout << "  coverage: nar/zero=" << nar_or_zero << " below_split=" << below_split
              << " all_lost=" << all_lost << " field_scale=" << field_scale
              << " mul_scale=" << mul_scale << " | exact=" << exact
              << " halfway=" << halfway << " inexact=" << inexact << std::endl;

    EXPECT_GT(nar_or_zero, 0u);
    EXPECT_GT(below_split, 0u) << "the `n_act < EXPMIN` fast path is never taken";
    EXPECT_GT(all_lost, 0u) << "the underflow-to-zero path is never taken";
    EXPECT_GT(field_scale, 0u) << "the exponent-field scaling path is never taken";
    EXPECT_GT(mul_scale, 0u) << "the multiplying scaling path is never taken";
    EXPECT_GT(exact, 0u) << "no exact rounding is exercised";
    EXPECT_GT(halfway, 0u) << "no exact ties are exercised";
    EXPECT_GT(inexact, 0u) << "no inexact rounding is exercised";
}

TEST(TestRoundScaled, RandomCoverageFloat) { check_round_scaled_coverage<float>(); }
TEST(TestRoundScaled, RandomCoverageDouble) { check_round_scaled_coverage<double>(); }

/// @brief Instantiates a check for both container types, both scaling
/// strategies, and both ways of reducing the scaled value to an integer.
#define ROUND_SCALED_TESTS(name, fn)                                            \
    TEST(TestRoundScaled, name##Float) { fn<float, true, false>(); }            \
    TEST(TestRoundScaled, name##Double) { fn<double, true, false>(); }          \
    TEST(TestRoundScaled, name##FloatScaleMul) { fn<float, false, false>(); }   \
    TEST(TestRoundScaled, name##DoubleScaleMul) { fn<double, false, false>(); } \
    TEST(TestRoundScaled, name##FloatFp) { fn<float, true, true>(); }           \
    TEST(TestRoundScaled, name##DoubleFp) { fn<double, true, true>(); }         \
    TEST(TestRoundScaled, name##FloatFpScaleMul) { fn<float, false, true>(); }  \
    TEST(TestRoundScaled, name##DoubleFpScaleMul) { fn<double, false, true>(); }

ROUND_SCALED_TESTS(Edges, check_round_scaled_edges)
ROUND_SCALED_TESTS(Halfway, check_round_scaled_halfway)
ROUND_SCALED_TESTS(DeepSubnormal, check_round_scaled_deep_subnormal)
ROUND_SCALED_TESTS(Random, check_round_scaled_random)
ROUND_SCALED_TESTS(RandomSubnormal, check_round_scaled_random_subnormal)

} // namespace
