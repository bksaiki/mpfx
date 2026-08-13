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
using mpfx::round_scaled::round_all_lost;

namespace {

// Iteration counts. These are unit tests, so the whole file is meant to stay in
// the tens of milliseconds; the exhaustive sweep below is strided for the same
// reason. Seeds are fixed, so a given count always covers the same inputs.
constexpr size_t N_RANDOM = 25'000;     // uniform encodings, times 8 modes
constexpr size_t N_SUBNORMAL = 10'000;  // subnormal encodings, times 8 modes
constexpr size_t N_ALL_LOST = 10'000;   // `n >= e` encodings, times 8 modes
constexpr size_t N_REFERENCE = 10'000;  // against `round_reference`
constexpr uint64_t EXHAUSTIVE_STRIDE = 65'537;

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

constexpr RM MODES[] = {
    RM::RNE, RM::RNA, RM::RTP, RM::RTN, RM::RTZ, RM::RAZ, RM::RTO, RM::RTE,
};

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
// round_all_lost
//

/// @brief Dispatches `round_all_lost` on a runtime rounding mode.
template <std::floating_point T>
bit_float<T> all_lost(bit_float<T> x, exp_t n, RM rm) {
    const exp_t e = x.e();
    switch (rm) {
    case RM::RNE: return round_all_lost<RM::RNE>(x, e, n);
    case RM::RNA: return round_all_lost<RM::RNA>(x, e, n);
    case RM::RTP: return round_all_lost<RM::RTP>(x, e, n);
    case RM::RTN: return round_all_lost<RM::RTN>(x, e, n);
    case RM::RTZ: return round_all_lost<RM::RTZ>(x, e, n);
    case RM::RAZ: return round_all_lost<RM::RAZ>(x, e, n);
    case RM::RTO: return round_all_lost<RM::RTO>(x, e, n);
    case RM::RTE: return round_all_lost<RM::RTE>(x, e, n);
    default: return x;
    }
}

/// @brief For split points at or above the normalized exponent, `round_all_lost`
/// must match `round_bits::round` in every rounding mode.
template <std::floating_point T>
void check_round_all_lost() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = N_ALL_LOST;

    std::mt19937_64 rng(0xA110C);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);

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
            // `n >= e >= e - p`, so the reference splits at `n` too
            const bit_float<T> want =
                mpfx::round_bits::round<mpfx::Flags::NO_FLAGS>(x, p, n, rm);
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

/// @brief Renders an optional subnormalization point for failure messages.
std::string describe_n(std::optional<exp_t> n) {
    return n.has_value() ? std::to_string(*n) : std::string("none");
}

/// @brief `round_scaled::round` must agree with `round_bits::round` bit for
/// bit, in every rounding mode, for one `(x, p, n)` triple.
template <std::floating_point T>
void expect_matches(bit_float<T> x, prec_t p, std::optional<exp_t> n, const std::string& label) {
    for (const RM rm : MODES) {
        const bit_float<T> want = mpfx::round_bits::round<mpfx::Flags::NO_FLAGS>(x, p, n, rm);
        const bit_float<T> got =
            mpfx::round_scaled::round<mpfx::Flags::NO_FLAGS>(x, p, n, rm);
        ASSERT_EQ(got.to_bits(), want.to_bits())
            << "round_scaled<" << rm_name(rm) << ">("
            << describe(x.to_float())
            << ", p=" << p << ", n=" << describe_n(n) << ") = " << describe(got.to_float())
            << ", want " << describe(want.to_float()) << " [" << label << "]";
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
template <std::floating_point T>
void check_round_scaled_edges() {
    using params_t = typename bit_float<T>::params_t;
    static constexpr prec_t P = params_t::P;

    const std::vector<prec_t> precs = {1, 2, P / 2, P - 1, P};
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
                expect_matches<T>(x, p, n, name);
            }
        }
    }
}


/// @brief Values sitting exactly at, just above, and just below the halfway
/// point for each target precision - the cases where the tie-breaking rules and
/// the parity of the last kept digit actually matter.
template <std::floating_point T>
void check_round_scaled_halfway() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr prec_t M = params_t::M;

    // `1 + 2^-p` splits at `n = -p` with a lost part of exactly `2^-p`. This stops
    // below `P` because the halfway digit sits at position `-p` in the mantissa
    // field, which has no such digit when `p == P`; the random tests cover that.
    for (prec_t p = 1; p < params_t::P; p++) {
        const uint_t one = params_t::IMPLICIT1;
        const auto at = static_cast<uint_t>(one | (static_cast<uint_t>(1) << (M - p)));
        expect_matches<T>(bit_float<T>(at), p, std::nullopt, "halfway, even");
        expect_matches<T>(
            bit_float<T>(static_cast<uint_t>(params_t::SMASK | at)), p, std::nullopt, "-halfway, even");

        // set the last kept digit to make it odd
        if (p >= 2) {
            const auto odd = static_cast<uint_t>(at | (static_cast<uint_t>(1) << (M - p + 1)));
            expect_matches<T>(bit_float<T>(odd), p, std::nullopt, "halfway, odd");
        }

        // one digit below the halfway digit puts the value off the tie
        if (p + 1 <= M) {
            const auto above = static_cast<uint_t>(at | (static_cast<uint_t>(1) << (M - p - 1)));
            const auto below = static_cast<uint_t>(one | (static_cast<uint_t>(1) << (M - p - 1)));
            expect_matches<T>(bit_float<T>(above), p, std::nullopt, "above halfway");
            expect_matches<T>(bit_float<T>(below), p, std::nullopt, "below halfway");
        }
    }
}


/// @brief Subnormal inputs shallow enough to keep a digit, which is where the
/// scaling has to bias `x` into the normal range first.
template <std::floating_point T>
void check_round_scaled_deep_subnormal() {
    using params_t = typename bit_float<T>::params_t;

    // a subnormal with a few digits, low enough that `2^-(n+1)` overflows
    for (exp_t e = params_t::EXPMIN + 1; e <= params_t::EXPMIN + 4; e++) {
        const T v = bit_float<T>::make_pow2(e).to_float()
                  + bit_float<T>::make_pow2(params_t::EXPMIN).to_float();
        for (prec_t p = 1; p < params_t::P; p++) {
            expect_matches<T>(bit_float<T>(v), p, std::nullopt, "deep subnormal");
            expect_matches<T>(bit_float<T>(-v), p, std::nullopt, "-deep subnormal");
            expect_matches<T>(bit_float<T>(v), p, params_t::EXPMIN - 1, "deep subnormal");
        }
    }
}


/// @brief Randomized differential test against `round_bits::round`.
template <std::floating_point T>
void check_round_scaled_random() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = N_RANDOM;

    std::mt19937_64 rng(0x5EEDU);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);
    std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMAX - 1);
    std::bernoulli_distribution has_n_dist(0.5);

    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        const prec_t p = prec_dist(rng);
        const std::optional<exp_t> n =
            has_n_dist(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;
        expect_matches<T>(x, p, n, "random");
    }
}

/// @brief Randomized differential test restricted to subnormal inputs.
///
/// Uniform bit patterns are subnormal only about once in five thousand draws for
/// `float` and once in two thousand for `double`, which leaves the biased scale
/// and the `n_act < EXPMIN` fast path barely exercised by
/// `check_round_scaled_random`. Drawing subnormals directly fixes that.
template <std::floating_point T>
void check_round_scaled_random_subnormal() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = N_SUBNORMAL;

    std::mt19937_64 rng(0x50BU);
    std::uniform_int_distribution<uint_t> mant_dist(1, static_cast<uint_t>(params_t::MMASK));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);
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
        expect_matches<T>(x, p, n, "random subnormal");
    }
}


/// @brief Which path through `round_scaled` an input takes, and what kind of
/// rounding it asks for.
struct PathCounts {
    size_t nar_or_zero = 0, below_split = 0, all_lost = 0, normal = 0, biased = 0;
    size_t exact = 0, halfway = 0, inexact = 0;
};

/// @brief Classifies one input the way `round_scaled` would branch on it.
template <std::floating_point T>
void classify(bit_float<T> x, prec_t p, std::optional<exp_t> n, PathCounts& c) {
    using params_t = typename bit_float<T>::params_t;

    if (x.is_nar() || x.is_zero()) {
        c.nar_or_zero++;
        return;
    }

    const exp_t e = x.e();
    const exp_t n_min = e - static_cast<exp_t>(p);
    const exp_t n_act = n.has_value() ? std::max(n_min, *n) : n_min;

    if (n_act < params_t::EXPMIN) {
        c.below_split++;
        return;
    }
    if (n_act >= e) {
        c.all_lost++;
        return;
    }

    if (x.ebits() != 0) {
        c.normal++;
    } else {
        c.biased++;   // subnormal: biased into the normal range before scaling
    }

    // classify the lost digits using the existing split
    const auto [hi, rs] = x.split_rs(n_act);
    (void) hi;
    if (rs == mpfx::RoundRS::EXACT) {
        c.exact++;
    } else if (rs == mpfx::RoundRS::EXACT_HALFWAY) {
        c.halfway++;
    } else {
        c.inexact++;
    }
}

/// @brief Confirms the randomized tests actually reach every path in
/// `round_scaled`, so that coverage cannot quietly disappear if the input
/// distributions or the iteration counts are ever retuned.
///
/// Both generators are classified together, because neither covers everything on
/// its own: uniform encodings are subnormal only about once in two thousand draws,
/// which at these counts leaves the biased path and the `n_act < EXPMIN` path to
/// the subnormal generator. The counts and seeds mirror
/// `check_round_scaled_random` and `check_round_scaled_random_subnormal`.
template <std::floating_point T>
void check_round_scaled_coverage() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    PathCounts c;
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);
    std::bernoulli_distribution coin(0.5);

    {   // the uniform generator
        std::mt19937_64 rng(0x5EEDU);
        std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
        std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMAX - 1);
        for (size_t i = 0; i < N_RANDOM; i++) {
            const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
            const prec_t p = prec_dist(rng);
            const std::optional<exp_t> n =
                coin(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;
            classify(x, p, n, c);
        }
    }

    {   // the subnormal generator
        std::mt19937_64 rng(0x50BU);
        std::uniform_int_distribution<uint_t> mant_dist(1, static_cast<uint_t>(params_t::MMASK));
        std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMIN);
        for (size_t i = 0; i < N_SUBNORMAL; i++) {
            uint_t bits = mant_dist(rng);
            if (coin(rng)) {
                bits |= static_cast<uint_t>(params_t::SMASK);
            }
            const prec_t p = prec_dist(rng);
            const std::optional<exp_t> n =
                coin(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;
            classify(bit_float<T>(bits), p, n, c);
        }
    }

    std::cout << "  coverage: nar/zero=" << c.nar_or_zero << " below_split=" << c.below_split
              << " all_lost=" << c.all_lost << " normal=" << c.normal
              << " biased=" << c.biased << " | exact=" << c.exact
              << " halfway=" << c.halfway << " inexact=" << c.inexact << std::endl;

    EXPECT_GT(c.nar_or_zero, 0u);
    EXPECT_GT(c.below_split, 0u) << "the `n_act < EXPMIN` fast path is never taken";
    EXPECT_GT(c.all_lost, 0u) << "the underflow-to-zero path is never taken";
    EXPECT_GT(c.normal, 0u) << "the unbiased path is never taken";
    EXPECT_GT(c.biased, 0u) << "the biased (subnormal) path is never taken";
    EXPECT_GT(c.exact, 0u) << "no exact rounding is exercised";
    EXPECT_GT(c.halfway, 0u) << "no exact ties are exercised";
    EXPECT_GT(c.inexact, 0u) << "no inexact rounding is exercised";
}

TEST(TestRoundScaled, RandomCoverageFloat) { check_round_scaled_coverage<float>(); }
TEST(TestRoundScaled, RandomCoverageDouble) { check_round_scaled_coverage<double>(); }

/// @brief Instantiates a check for both container types.
#define ROUND_SCALED_TESTS(name, fn)                        \
    TEST(TestRoundScaled, name##Float) { fn<float>(); }     \
    TEST(TestRoundScaled, name##Double) { fn<double>(); }

ROUND_SCALED_TESTS(Edges, check_round_scaled_edges)
ROUND_SCALED_TESTS(Halfway, check_round_scaled_halfway)
ROUND_SCALED_TESTS(DeepSubnormal, check_round_scaled_deep_subnormal)
ROUND_SCALED_TESTS(Random, check_round_scaled_random)
ROUND_SCALED_TESTS(RandomSubnormal, check_round_scaled_random_subnormal)

//
// status flags
//

/// @brief Snapshots the status flags that rounding is responsible for.
uint32_t flag_snapshot() {
    uint32_t w = 0;
    if (mpfx::flags.tiny_before_rounding())       { w |= mpfx::Flags::TINY_BEFORE_ROUNDING_FLAG; }
    if (mpfx::flags.tiny_after_rounding())        { w |= mpfx::Flags::TINY_AFTER_ROUNDING_FLAG; }
    if (mpfx::flags.underflow_before_rounding())  { w |= mpfx::Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG; }
    if (mpfx::flags.underflow_after_rounding())   { w |= mpfx::Flags::UNDERFLOW_AFTER_ROUNDING_FLAG; }
    if (mpfx::flags.inexact())                    { w |= mpfx::Flags::INEXACT_FLAG; }
    if (mpfx::flags.carry())                      { w |= mpfx::Flags::CARRY_FLAG; }
    return w;
}

/// @brief Names the flags in a mask, for failure messages.
std::string describe_flags(uint32_t w) {
    if (w == 0) {
        return "none";
    }
    std::string out;
    const std::pair<uint32_t, const char*> names[] = {
        {mpfx::Flags::TINY_BEFORE_ROUNDING_FLAG, "tiny_before"},
        {mpfx::Flags::TINY_AFTER_ROUNDING_FLAG, "tiny_after"},
        {mpfx::Flags::UNDERFLOW_BEFORE_ROUNDING_FLAG, "underflow_before"},
        {mpfx::Flags::UNDERFLOW_AFTER_ROUNDING_FLAG, "underflow_after"},
        {mpfx::Flags::INEXACT_FLAG, "inexact"},
        {mpfx::Flags::CARRY_FLAG, "carry"},
    };
    for (const auto& [bit, nm] : names) {
        if (w & bit) {
            if (!out.empty()) { out += "|"; }
            out += nm;
        }
    }
    return out;
}

/// @brief `round_scaled::round` must raise exactly the flags `round_bits::round`
/// raises, and
/// return the same value, for one `(x, p, n)` triple in every rounding mode.
template <std::floating_point T>
void expect_flags_match(bit_float<T> x, prec_t p, std::optional<exp_t> n, const std::string& label) {
    for (const RM rm : MODES) {
        mpfx::flags.reset();
        const bit_float<T> want = mpfx::round_bits::round<mpfx::Flags::ALL_FLAGS>(x, p, n, rm);
        const uint32_t want_flags = flag_snapshot();

        mpfx::flags.reset();
        const bit_float<T> got = mpfx::round_scaled::round<mpfx::Flags::ALL_FLAGS>(x, p, n, rm);
        const uint32_t got_flags = flag_snapshot();

        ASSERT_EQ(got.to_bits(), want.to_bits())
            << "value: round_scaled<" << rm_name(rm) << ">(" << describe(x.to_float())
            << ", p=" << p << ", n=" << describe_n(n) << ") [" << label << "]";
        ASSERT_EQ(got_flags, want_flags)
            << "flags: round_scaled<" << rm_name(rm) << ">(" << describe(x.to_float())
            << ", p=" << p << ", n=" << describe_n(n) << ") raised {" << describe_flags(got_flags)
            << "}, round raised {" << describe_flags(want_flags) << "} [" << label << "]";
    }
}

/// @brief Flags over the interesting values, crossed with the interesting `(p, n)`.
template <std::floating_point T>
void check_flags_edges() {
    using params_t = typename bit_float<T>::params_t;
    static constexpr prec_t P = params_t::P;
    const std::vector<prec_t> precs = {1, 2, P / 2, P - 1, P};
    // The tininess flags are defined against `emin = n + p`, the target format's
    // smallest normalized exponent, so `n` is kept low enough that `emin` stays
    // representable in the container. Above that both implementations return
    // artifacts of infinity arithmetic - `2^emin` is not a number the container
    // holds - and the configuration is meaningless anyway.
    const std::vector<std::optional<exp_t>> ns = {
        std::nullopt, params_t::EXPMIN - 1, params_t::EXPMIN + 1,
        params_t::EMIN - static_cast<exp_t>(P), -1,
        params_t::EMAX - static_cast<exp_t>(P),
    };
    for (const auto& [name, x] : edge_values<T>()) {
        for (const prec_t p : precs) {
            for (const auto& n : ns) {
                expect_flags_match<T>(x, p, n, name);
            }
        }
    }
}

/// @brief Flags over uniform and subnormal random encodings.
template <std::floating_point T>
void check_flags_random() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    std::mt19937_64 rng(0xF1A65U);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<uint_t> mant_dist(1, static_cast<uint_t>(params_t::MMASK));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);
    std::bernoulli_distribution coin(0.5);

    // `n` is drawn near the bottom of the format, since the tininess flags only
    // engage when `n` is present and `x` falls below `2^(n+p)`
    std::uniform_int_distribution<exp_t> n_low(params_t::EXPMIN - 1, params_t::EMIN + 4);
    std::uniform_int_distribution<exp_t> n_any(params_t::EXPMIN - 1,
                                              params_t::EMAX - static_cast<exp_t>(params_t::P));

    for (size_t i = 0; i < N_RANDOM; i++) {
        const bool subnormal = coin(rng);
        const bit_float<T> x(static_cast<uint_t>(subnormal ? mant_dist(rng) : bits_dist(rng)));
        const prec_t p = prec_dist(rng);
        const std::optional<exp_t> n =
            coin(rng) ? std::optional<exp_t>(coin(rng) ? n_low(rng) : n_any(rng)) : std::nullopt;
        expect_flags_match<T>(x, p, n, subnormal ? "random subnormal" : "random");
    }
}

ROUND_SCALED_TESTS(FlagsEdges, check_flags_edges)
ROUND_SCALED_TESTS(FlagsRandom, check_flags_random)

//
// a second, independent oracle
//

/// @brief Rounds `x` through `round_reference`, the original integer-significand
/// implementation, which shares no code with either of the other two.
///
/// This matters as a third witness: the tests above compare `round_scaled::round`
/// against `round_bits::round`, and the `(m, exp)` overload of the public and the integer overload of the public
/// `round` reaches `round_reference::round_finalize`, which is neither of them.
///
/// Returns nothing for the cases it cannot express: NaN and
/// infinity have no `(m, exp)` form, signed zero loses its sign through an
/// integer significand, and a result outside the range of `T` comes back as a
/// finite `double` rather than saturating.
template <std::floating_point T>
std::optional<T> reference_round(bit_float<T> x, prec_t p, std::optional<exp_t> n, RM rm) {
    using limits = std::numeric_limits<T>;
    if (x.is_nar() || x.is_zero()) {
        return std::nullopt;
    }

    const auto [s, exp, c] = x.unpack();
    const auto mag = static_cast<int64_t>(c);
    const double r = mpfx::round<mpfx::Flags::NO_FLAGS>(s ? -mag : mag, exp, p, n, rm);
    if (std::abs(r) > static_cast<double>(limits::max())) {
        return std::nullopt;
    }
    return static_cast<T>(r);
}

/// @brief The carry flag for subnormal inputs.
///
/// A subnormal `x` can carry onto a larger *subnormal* power of two, where the
/// exponent field does not change and the mantissa field is non-zero. Both new
/// implementations once missed that, so comparing them against each other cannot
/// catch it.
///
/// The reference is the one `TestFlags.TestCarryFlag` uses - `x != 0 && y != 0 &&
/// ye > xe && xe >= emin` - applied to the returned value, whose correctness the
/// tests above establish independently. That test never reaches a container
/// subnormal (it draws exponents in `[-4, 4]`), which is why it missed this. Here
/// `n` is absent, so `emin` is unbounded below and `xe >= emin` holds; `p >= 1`
/// with no subnormalization keeps at least one digit, so `y != 0` holds too.
/// `round_reference` is checked to agree, as a second witness.
template <std::floating_point T>
void check_carry_subnormal() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;

    std::mt19937_64 rng(0xCA881);
    std::uniform_int_distribution<uint_t> mant_dist(1, static_cast<uint_t>(params_t::MMASK));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);
    std::bernoulli_distribution coin(0.5);

    for (size_t i = 0; i < N_SUBNORMAL; i++) {
        uint_t bits = mant_dist(rng);
        if (coin(rng)) {
            bits |= static_cast<uint_t>(params_t::SMASK);
        }
        const bit_float<T> x(bits);
        const prec_t p = prec_dist(rng);
        const auto [s, exp, c] = x.unpack();
        const auto mag = static_cast<int64_t>(c);

        for (const RM rm : MODES) {
            mpfx::flags.reset();
            const bit_float<T> r =
                mpfx::round_bits::round<mpfx::Flags::ALL_FLAGS>(x, p, std::nullopt, rm);
            const bool got_round = mpfx::flags.carry();
            const bool want = r.e() > x.e();

            mpfx::flags.reset();
            mpfx::round_scaled::round<mpfx::Flags::ALL_FLAGS>(x, p, std::nullopt, rm);
            const bool got_scaled = mpfx::flags.carry();

            mpfx::flags.reset();
            mpfx::round<mpfx::Flags::ALL_FLAGS>(s ? -mag : mag, exp, p, std::nullopt, rm);
            const bool want_reference = mpfx::flags.carry();

            const std::string what = std::string("(") + describe(x.to_float())
                                   + ", p=" + std::to_string(p) + ") in " + rm_name(rm)
                                   + ", result " + describe(r.to_float());
            ASSERT_EQ(want_reference, want) << "carry: `round_reference` disagrees with the "
                                            "definition for " << what;
            ASSERT_EQ(got_round, want) << "carry: round " << what;
            ASSERT_EQ(got_scaled, want) << "carry: round_scaled " << what;
        }
    }
}

ROUND_SCALED_TESTS(CarrySubnormal, check_carry_subnormal)

/// @brief Randomized differential test against `round_reference`.
template <std::floating_point T>
void check_round_scaled_reference() {
    using params_t = typename bit_float<T>::params_t;
    using uint_t = typename bit_float<T>::uint_t;
    static constexpr size_t N = N_REFERENCE;

    std::mt19937_64 rng(0x1EACADEU);
    std::uniform_int_distribution<uint_t> bits_dist(0, ~static_cast<uint_t>(0));
    std::uniform_int_distribution<prec_t> prec_dist(1, params_t::P);
    std::uniform_int_distribution<exp_t> n_dist(params_t::EXPMIN - 1, params_t::EMAX - 1);
    std::bernoulli_distribution coin(0.5);

    size_t checked = 0;
    for (size_t i = 0; i < N; i++) {
        const bit_float<T> x(static_cast<uint_t>(bits_dist(rng)));
        const prec_t p = prec_dist(rng);
        const std::optional<exp_t> n =
            coin(rng) ? std::optional<exp_t>(n_dist(rng)) : std::nullopt;

        for (const RM rm : MODES) {
            const std::optional<T> want = reference_round(x, p, n, rm);
            if (!want.has_value()) {
                continue;
            }

            checked++;
            const bit_float<T> got =
                mpfx::round_scaled::round<mpfx::Flags::NO_FLAGS>(x, p, n, rm);
            ASSERT_TRUE(same_bits(got.to_float(), *want))
                << "round_scaled<" << rm_name(rm) << ">(" << describe(x.to_float()) << ", p=" << p << ", n=" << describe_n(n) << ") = "
                << describe(got.to_float()) << ", `round_reference` says " << describe(*want);
        }
    }

    EXPECT_GT(checked, N) << "too few cases were comparable against `round_reference`";
}

ROUND_SCALED_TESTS(Reference, check_round_scaled_reference)

//
// exhaustive sweeps over the `float` encoding
//

/// @brief Compares every `stride`-th `float` encoding against `round`, for a
/// handful of representative configurations in every rounding mode.
///
/// The comparison is hand-rolled rather than going through `expect_matches` so
/// that no gtest machinery runs in the inner loop; only a mismatch reports.
void check_exhaustive_float(uint64_t stride) {
    using params_t = typename bit_float<float>::params_t;
    struct Cfg { prec_t p; std::optional<exp_t> n; };
    const Cfg cfgs[] = {
        {12, std::nullopt},                                   // precision only
        {params_t::P - 1, params_t::EMIN - static_cast<exp_t>(params_t::P)}, // IEEE 754 f32
        {4, -10},                                             // narrow, subnormalizing
        {params_t::P, std::nullopt},                          // the container's own precision
    };

    for (const auto& cfg : cfgs) {
        for (const RM rm : MODES) {
            for (uint64_t i = 0; i < (1ULL << 32); i += stride) {
                const bit_float<float> x(static_cast<uint32_t>(i));
                const bit_float<float> want =
                    mpfx::round_bits::round<mpfx::Flags::NO_FLAGS>(x, cfg.p, cfg.n, rm);
                const bit_float<float> got =
                    mpfx::round_scaled::round<mpfx::Flags::NO_FLAGS>(
                        x, cfg.p, cfg.n, rm);
                if (got.to_bits() != want.to_bits()) [[unlikely]] {
                    FAIL() << "round_scaled<" << rm_name(rm)
                           << ">(" << describe(x.to_float()) << ", p=" << cfg.p
                           << ", n=" << describe_n(cfg.n) << ") = " << describe(got.to_float())
                           << ", want " << describe(want.to_float());
                }
            }
        }
    }
}

// A stride keeps the default suite quick while still covering the whole
// encoding uniformly, including every exponent and both signs.
TEST(TestRoundScaled, ExhaustiveStridedFloat) { check_exhaustive_float(EXHAUSTIVE_STRIDE); }

} // namespace
