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
    EXPECT_EQ(ctx.overflow(), mpfx::OverflowMode::OVERFLOW);
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
    EXPECT_EQ(ctx.overflow(), mpfx::OverflowMode::SATURATE);
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
}
