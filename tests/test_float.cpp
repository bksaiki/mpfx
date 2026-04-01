#include <random>

#include <gtest/gtest.h>
#include <mpfx.hpp>
#include <mpfx/float.hpp>

TEST(TestBitFloat, TestF32Construct) {
    mpfx::bit_float<float> bf1; // default constructor
    EXPECT_EQ(bf1.to_bits(), 0u);
    EXPECT_EQ(bf1.to_float(), 0.0f);

    float value = 3.14f;
    mpfx::bit_float<float> bf2(value); // construct from float
    EXPECT_EQ(bf2.to_bits(), std::bit_cast<uint32_t>(value));
    EXPECT_EQ(bf2.to_float(), value);

    mpfx::bit_float<float>::uint_t raw_bits = 0x4048f5c3; // raw bits for 3.14f
    mpfx::bit_float<float> bf3(raw_bits); // construct from raw bits
    EXPECT_EQ(bf3.to_bits(), raw_bits);
    EXPECT_EQ(bf3.to_float(), value);
}

TEST(TestBitFloat, TestF64Construct) {
    mpfx::bit_float<double> bf1; // default constructor
    EXPECT_EQ(bf1.to_bits(), 0ull);
    EXPECT_EQ(bf1.to_float(), 0.0);

    double value = 3.14;
    mpfx::bit_float<double> bf2(value); // construct from double
    EXPECT_EQ(bf2.to_bits(), std::bit_cast<uint64_t>(value));
    EXPECT_EQ(bf2.to_float(), value);

    mpfx::bit_float<double>::uint_t raw_bits = 0x40091eb851eb851full; // raw bits for 3.14
    mpfx::bit_float<double> bf3(raw_bits); // construct from raw bits
    EXPECT_EQ(bf3.to_bits(), raw_bits);
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
    EXPECT_EQ(e, -149); // smallest subnormal has exponent -149
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

TEST(TestBitFloat, TestUnpack) {
    float value = 3.14f;
    mpfx::bit_float<float> bf(value);
    auto [s, exp, c] = bf.unpack();
    EXPECT_EQ(s, 0); // 3.14 is positive
    EXPECT_EQ(exp, -22); // 3.14 has unnormalized exponent -22
    EXPECT_EQ(c, 0xc8f5c3); // significand bits for 3.14f
}

TEST(TestBitFloat, TestSplit) {
    float value = 1.5f;
    mpfx::bit_float<float> bf(value);
    auto split = bf.split(-1); // split at the subnormal
    EXPECT_EQ(split.first.to_float(), 1.0f); // high part should be 1.0
    EXPECT_EQ(split.second.to_float(), 0.5f); // low part should be 0.5

    mpfx::bit_float<float>::uint_t raw_bits = 0x00000003; // small subnormal number
    bf = mpfx::bit_float<float>(raw_bits);
    split = bf.split(-149);
    EXPECT_EQ(split.first.to_bits(), 0x00000002); // high part
    EXPECT_EQ(split.second.to_bits(), 0x00000001); // low part
}

TEST(TestBitFloat, TestSplitRandom) {
    static constexpr size_t N = 1'000'000;

    using FP32 = typename mpfx::float_params<float>::params;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());
    std::uniform_int_distribution<uint32_t> dist(0, 0xffffffff);

    for (size_t i = 0; i < N; i++) {
        // generate a random bitstring
        uint32_t raw_bits = dist(rng);
        mpfx::bit_float<float> bf(raw_bits);

        // check if the value is NaN or Inf
        if (bf.is_nar()) {
            continue; // skip NaN and Inf values
        }

        // generate a random digit position
        std::uniform_int_distribution<mpfx::exp_t> n_dist(FP32::EXPMIN, FP32::EMAX);
        mpfx::exp_t n = n_dist(rng);

        // split the bit_float at position n
        auto [high, low] = bf.split(n);

        // check properties
        EXPECT_EQ(high.s(), bf.s()); // sign should be the same
        EXPECT_EQ(bf.to_float(), high.to_float() + low.to_float());
    }
}
