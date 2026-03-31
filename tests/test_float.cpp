#include <gtest/gtest.h>
#include <mpfx.hpp>
#include <mpfx/float.hpp>

TEST(TestBitFloat, TestF32Construct) {
    mpfx::bit_float<float> bf1; // default constructor
    EXPECT_EQ(bf1.bits(), 0u);
    EXPECT_EQ(bf1.to_float(), 0.0f);

    float value = 3.14f;
    mpfx::bit_float<float> bf2(value); // construct from float
    EXPECT_EQ(bf2.bits(), std::bit_cast<uint32_t>(value));
    EXPECT_EQ(bf2.to_float(), value);

    mpfx::bit_float<float>::uint_t raw_bits = 0x4048f5c3; // raw bits for 3.14f
    mpfx::bit_float<float> bf3(raw_bits); // construct from raw bits
    EXPECT_EQ(bf3.bits(), raw_bits);
    EXPECT_EQ(bf3.to_float(), value);
}

TEST(TestBitFloat, TestF64Construct) {
    mpfx::bit_float<double> bf1; // default constructor
    EXPECT_EQ(bf1.bits(), 0ull);
    EXPECT_EQ(bf1.to_float(), 0.0);

    double value = 3.14;
    mpfx::bit_float<double> bf2(value); // construct from double
    EXPECT_EQ(bf2.bits(), std::bit_cast<uint64_t>(value));
    EXPECT_EQ(bf2.to_float(), value);

    mpfx::bit_float<double>::uint_t raw_bits = 0x40091eb851eb851full; // raw bits for 3.14
    mpfx::bit_float<double> bf3(raw_bits); // construct from raw bits
    EXPECT_EQ(bf3.bits(), raw_bits);
    EXPECT_EQ(bf3.to_float(), value);
}

TEST(TestBitFloat, TestP) {
    float value = 1.0f;
    mpfx::bit_float<float> bf(value);
    EXPECT_EQ(bf.p(), 24); // 1.0 is a normal number with full precision

    mpfx::bit_float<float>::uint_t raw_bits = 0x00000001; // smallest positive subnormal number
    bf = mpfx::bit_float<float>(raw_bits);
    EXPECT_EQ(bf.p(), 1);
}

TEST(TestBitFloat, TestE) {
    float value = 1.0f;
    mpfx::bit_float<float> bf(value);
    int e = bf.e();
    EXPECT_EQ(e, 0); // 1.0 has exponent 0 (after bias)

    value = 0.5f;
    bf = mpfx::bit_float<float>(value);
    e = bf.e();
    EXPECT_EQ(e, -1); // 0.5 has exponent -1 (after bias)

    value = 2.0f;
    bf = mpfx::bit_float<float>(value);
    e = bf.e();
    EXPECT_EQ(e, 1); // 2.0 has exponent 1 (after bias)

    mpfx::bit_float<float>::uint_t raw_bits = 0x00000001; // smallest positive subnormal number
    bf = mpfx::bit_float<float>(raw_bits);
    e = bf.e();
    EXPECT_EQ(e, -126); // smallest subnormal has exponent -126
}

TEST(TestBitFloat, TestExp) {
    float value = 1.0f;
    mpfx::bit_float<float> bf(value);
    int exp = bf.exp();
    EXPECT_EQ(exp, -23); // 1.0 has unnormalized exponent -23

    value = 0.5f;
    bf = mpfx::bit_float<float>(value);
    exp = bf.exp();
    EXPECT_EQ(exp, -24); // 0.5 has unnormalized exponent -24

    value = 2.0f;
    bf = mpfx::bit_float<float>(value);
    exp = bf.exp();
    EXPECT_EQ(exp, -22); // 2.0 has unnormalized exponent -22

    mpfx::bit_float<float>::uint_t raw_bits = 0x00000001; // smallest positive subnormal number
    bf = mpfx::bit_float<float>(raw_bits);
    exp = bf.exp();
    EXPECT_EQ(exp, -149); // smallest subnormal has unnormalized exponent -149
}
