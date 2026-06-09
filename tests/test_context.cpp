#include <mpfx.hpp>
#include <gtest/gtest.h>

using namespace mpfx;

TEST(Context, TestMPContext) {
    const MPContext ctx(5, mpfx::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 5);
    EXPECT_EQ(ctx.rm(), mpfx::RM::RNE);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 7);
    // rounding
    EXPECT_EQ(ctx.round(33.0), 32.0);
}

TEST(Context, TestMPSContext) {
    const MPSContext ctx(5, -5, mpfx::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 5);
    EXPECT_EQ(ctx.emin(), -5);
    EXPECT_EQ(ctx.rm(), mpfx::RM::RNE);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 7);
    EXPECT_EQ(ctx.n(), -10);
    // rounding
    EXPECT_EQ(ctx.round(33.0), 32.0);
    EXPECT_EQ(ctx.round(0.00048828125), 0.0);
}

TEST(Context, TestMPBContext) {
    const MPBContext ctx(5, -5, 62.0, mpfx::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 5);
    EXPECT_EQ(ctx.emin(), -5);
    EXPECT_EQ(ctx.rm(), mpfx::RM::RNE);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 7);
    EXPECT_EQ(ctx.n(), -10);
    // rounding
    EXPECT_EQ(ctx.round(33.0), 32.0);
    EXPECT_EQ(ctx.round(0.00048828125), 0.0);
    // overflow handling
    EXPECT_EQ(ctx.round(60.0), 60.0); // below maxval
    EXPECT_EQ(ctx.round(62.0), 62.0); // exact maxval
    EXPECT_EQ(ctx.round(63.0), std::numeric_limits<double>::infinity()); // rounds to infinity
    EXPECT_EQ(ctx.round(64.0), std::numeric_limits<double>::infinity()); // rounds to infinity
}

TEST(Context, TestIEEE754ContextFP32) {
    const IEEE754Context ctx(8, 32, mpfx::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 24);
    EXPECT_EQ(ctx.emin(), -126);
    EXPECT_EQ(ctx.emax(), 127);
    EXPECT_EQ(ctx.rm(), mpfx::RM::RNE);
    EXPECT_EQ(ctx.maxval(), 3.4028234663852886e+38);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 26);
}

TEST(Context, TestIEEE754ContextFP16) {
    const IEEE754Context ctx(5, 16, mpfx::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 11);
    EXPECT_EQ(ctx.emin(), -14);
    EXPECT_EQ(ctx.emax(), 15);
    EXPECT_EQ(ctx.rm(), mpfx::RM::RNE);
    EXPECT_EQ(ctx.maxval(), 65504.0);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 13);
}

TEST(Context, TestEFloatContextFP16) {
    // an IEEE-754-style format with infinities matches IEEE754Context(5, 16)
    const EFloatContext ctx(5, 16, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.prec(), 11);
    EXPECT_EQ(ctx.emin(), -14);
    EXPECT_EQ(ctx.emax(), 15);
    EXPECT_EQ(ctx.n(), -25);
    EXPECT_EQ(ctx.maxval(), 65504.0);
    EXPECT_EQ(ctx.overflow(), mpfx::OverflowMode::TO_INF);
    // overflow rounds to infinity
    EXPECT_EQ(ctx.round(1e30), std::numeric_limits<double>::infinity());
}

TEST(Context, TestEFloatContextE5M2) {
    // OCP FP8 E5M2: es=5, nbits=8, infinities, IEEE 754 NaNs
    const EFloatContext ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.prec(), 3);
    EXPECT_EQ(ctx.emax(), 15);
    EXPECT_EQ(ctx.maxval(), 57344.0);
    EXPECT_EQ(ctx.round(57344.0), 57344.0);
    EXPECT_EQ(ctx.round(1e30), std::numeric_limits<double>::infinity());
}

TEST(Context, TestEFloatContextE4M3) {
    // OCP FP8 E4M3: es=4, nbits=8, no infinities, MAX_VAL NaN (S.1111.111)
    const EFloatContext ctx(4, 8, false, EFloatNanKind::MAX_VAL, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.prec(), 4);
    EXPECT_EQ(ctx.maxval(), 448.0);
    EXPECT_EQ(ctx.round(448.0), 448.0);
    // infinities are disabled but NaNs exist: overflow remaps to NaN
    EXPECT_TRUE(std::isnan(ctx.round(1e30)));
    EXPECT_TRUE(std::isnan(ctx.round(std::numeric_limits<double>::infinity())));
}

TEST(Context, TestEFloatContextSaturate) {
    // no infinities and no NaNs: overflow saturates to maxval
    const EFloatContext ctx(4, 8, false, EFloatNanKind::NONE, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.overflow(), mpfx::OverflowMode::TO_MAXVAL);
    EXPECT_EQ(ctx.round(1e30), ctx.maxval());
    // NaN input is unrepresentable: maps to maxval
    EXPECT_EQ(ctx.round(std::numeric_limits<double>::quiet_NaN()), ctx.maxval());
}

TEST(Context, TestEFloatContextValid) {
    // constructor is usable in a constant expression
    constexpr EFloatContext ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    static_assert(ctx.is_valid());
    static_assert(ctx.prec() == 3);

    // valid formats
    EXPECT_TRUE((EFloatContext(4, 8, false, EFloatNanKind::MAX_VAL, 0, mpfx::RM::RNE).is_valid()));
    EXPECT_TRUE((EFloatContext(0, 8, false, EFloatNanKind::NONE, 0, mpfx::RM::RNE).is_valid()));

    // invalid: es >= nbits, or p < 2
    EXPECT_FALSE((EFloatContext(8, 8, false, EFloatNanKind::NONE, 0, mpfx::RM::RNE).is_valid()));
    EXPECT_FALSE((EFloatContext(7, 8, false, EFloatNanKind::NONE, 0, mpfx::RM::RNE).is_valid()));
    // invalid: IEEE 754 NaNs require an exponent field
    EXPECT_FALSE((EFloatContext(0, 8, false, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE).is_valid()));
}

TEST(Context, TestEFloatContextEOffset) {
    // shifting the exponent range scales maxval by 2^eoffset
    const EFloatContext base(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    const EFloatContext shifted(5, 8, true, EFloatNanKind::IEEE_754, -2, mpfx::RM::RNE);
    EXPECT_EQ(shifted.emax(), base.emax() - 2);
    EXPECT_EQ(shifted.emin(), base.emin() - 2);
    EXPECT_EQ(*shifted.maxval(), *base.maxval() / 4.0);
    // the shifted format overflows at a smaller magnitude
    EXPECT_EQ(shifted.maxval(), 14336.0);
    EXPECT_EQ(shifted.round(14336.0), 14336.0);
    EXPECT_EQ(shifted.round(57344.0), std::numeric_limits<double>::infinity());
}

TEST(Context, TestEFloatContextRounding) {
    // E5M2: precision is 3 bits, so the binade [8, 16) has ulp 2
    const EFloatContext ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.round(8.0), 8.0);    // exact
    EXPECT_EQ(ctx.round(9.0), 8.0);    // halfway 8/10, ties to even
    EXPECT_EQ(ctx.round(11.0), 12.0);  // halfway 10/12, ties to even
    EXPECT_EQ(ctx.round(10.0), 10.0);  // exact
}

TEST(Context, TestEFloatContextSubnormal) {
    // E5M2: expmin = emin - p + 1 = -16, so the smallest subnormal is 2^-16
    const EFloatContext ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.round(std::ldexp(1.0, -16)), std::ldexp(1.0, -16)); // min subnormal
    EXPECT_EQ(ctx.round(std::ldexp(1.0, -17)), 0.0);                  // underflows to zero
    EXPECT_EQ(ctx.round(std::ldexp(-1.0, -17)), 0.0);                 // signed zero magnitude
    EXPECT_TRUE(std::signbit(ctx.round(std::ldexp(-1.0, -17))));      // ... keeps its sign
}

TEST(Context, TestEFloatContextRoundingMode) {
    // round-toward-zero overflow saturates to maxval even when inf is enabled
    const EFloatContext ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RTZ);
    EXPECT_EQ(ctx.round(1e30), 57344.0);
    EXPECT_EQ(ctx.round(-1e30), -57344.0);
}

TEST(Context, TestEFloatContextSignedOverflow) {
    // sign is preserved through every overflow/fixup path
    const EFloatContext inf_ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    EXPECT_EQ(inf_ctx.round(-1e30), -std::numeric_limits<double>::infinity());

    const EFloatContext nan_ctx(4, 8, false, EFloatNanKind::MAX_VAL, 0, mpfx::RM::RNE);
    const double n = nan_ctx.round(-1e30);
    EXPECT_TRUE(std::isnan(n));
    EXPECT_TRUE(std::signbit(n));

    const EFloatContext sat_ctx(4, 8, false, EFloatNanKind::NONE, 0, mpfx::RM::RNE);
    EXPECT_EQ(sat_ctx.maxval(), 480.0); // no pattern reserved for Inf/NaN
    EXPECT_EQ(sat_ctx.round(-1e30), -480.0);
}

TEST(Context, TestEFloatContextNegZero) {
    // NEG_ZERO reuses the -0 slot for NaN, so a NaN exists: overflow without
    // infinities remaps to NaN (unlike NONE, which saturates).
    const EFloatContext ctx(4, 8, false, EFloatNanKind::NEG_ZERO, 0, mpfx::RM::RNE);
    EXPECT_EQ(ctx.maxval(), 480.0); // same binade as NONE
    EXPECT_EQ(ctx.overflow(), mpfx::OverflowMode::TO_INF);
    EXPECT_TRUE(std::isnan(ctx.round(1e30)));
}

TEST(Context, TestEFloatContextIntOverload) {
    // the round(m, exp) overload also applies the fixup
    const EFloatContext inf_ctx(5, 8, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    EXPECT_EQ(inf_ctx.round(int64_t{3}, 0), 3.0);                            // 3 * 2^0
    EXPECT_EQ(inf_ctx.round(int64_t{1}, 100),                               // 2^100 overflows
              std::numeric_limits<double>::infinity());

    const EFloatContext nan_ctx(4, 8, false, EFloatNanKind::MAX_VAL, 0, mpfx::RM::RNE);
    EXPECT_TRUE(std::isnan(nan_ctx.round(int64_t{1}, 100)));
}

TEST(Context, TestEFloatContextMatchesIEEE754) {
    // an IEEE-754-style EFloat format must round identically to IEEE754Context
    const IEEE754Context ref(8, 32, mpfx::RM::RNE);
    const EFloatContext ctx(8, 32, true, EFloatNanKind::IEEE_754, 0, mpfx::RM::RNE);
    const double values[] = {
        0.0, 1.0, -1.0, 0.1, 3.14159265358979, -2.5,
        1.0000000001, std::ldexp(1.0, -140), // fp32 subnormal range
        1e40, -1e40,                          // overflow to infinity
    };
    for (double v : values) {
        EXPECT_EQ(ctx.round(v), ref.round(v)) << "mismatch at v=" << v;
    }
}
