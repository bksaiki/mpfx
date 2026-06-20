#include <mpfr.h>
#include <random>

#include <mpfx.hpp>
#include <gtest/gtest.h>

static mpfr_rnd_t cvt_rm(mpfx::RM rm) {
    switch (rm) {
        case mpfx::RM::RNE:
            return MPFR_RNDN;
        case mpfx::RM::RTP:
            return MPFR_RNDU;
        case mpfx::RM::RTN:
            return MPFR_RNDD;
        case mpfx::RM::RTZ:
            return MPFR_RNDZ;
        case mpfx::RM::RAZ:
            return MPFR_RNDA;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

static double ref_add(double x, double y, int p, mpfx::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_add(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_sub(double x, double y, int p, mpfx::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_sub(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_mul(double x, double y, int p, mpfx::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_mul(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_div(double x, double y, int p, mpfx::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_div(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_sqrt(double x, int p, mpfx::RM rm) {
    mpfr_t mx, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_sqrt(mr, mx, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(mr);
    return r;
}

static double ref_fma(double x, double y, double z, int p, mpfx::RM rm) {
    mpfr_t mx, my, mz, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mz, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_set_d(mz, z, MPFR_RNDN);
    mpfr_fma(mr, mx, my, mz, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mz);
    mpfr_clear(mr);
    return r;
}


TEST(OpsFloat, TestAddUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_add(x, y, p, rm);
                const double z = mpfx::add(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestAddEFTUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_add(x, y, p, rm);
                const double z = mpfx::add<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestSubUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_sub(x, y, p, rm);
                const double z = mpfx::sub(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestMulUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_mul(x, y, p, rm);
                const double z = mpfx::mul(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestMulEFTUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_mul(x, y, p, rm);
                const double z = mpfx::mul<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestDivUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                // skip division by values close to zero
                if (std::abs(y) < 1e-10) continue;

                const double z_ref = ref_div(x, y, p, rm);
                const double z = mpfx::div(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestDivEFTUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_div(x, y, p, rm);
                const double z = mpfx::div<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestSqrtUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [0, 1]
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);

                const double z_ref = ref_sqrt(x, p, rm);
                const double z = mpfx::sqrt(x, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestSqrtEFTUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [0, 1]
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);

                const double z_ref = ref_sqrt(x, p, rm);
                const double z = mpfx::sqrt<mpfx::Engine::EFT>(x, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestFmaUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);
                const double z = dist(rng);

                const double w_ref = ref_fma(x, y, z, p, rm);
                const double w = mpfx::fma(x, y, z, ctx);
                EXPECT_EQ(w_ref, w);
            }
        }
    }
}

TEST(OpsFloat, TestFmaEFTUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<mpfx::RM> rounding_modes = {
        mpfx::RM::RNE,
        mpfx::RM::RTP,
        mpfx::RM::RTN,
        mpfx::RM::RTZ,
        mpfx::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const mpfx::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);
                const double z = dist(rng);

                const double w_ref = ref_fma(x, y, z, p, rm);
                const double w = mpfx::fma<mpfx::Engine::EFT>(x, y, z, ctx);
                EXPECT_EQ(w_ref, w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `float` (single-precision) variants. Only the EFT engine is type-generic, so
// these exercise the `float` round/EFT pipeline end-to-end. A `float` input is
// an exact `double`, and for context precision p <= 22 the EFT result is the
// correctly-rounded p-bit value regardless of container, so the existing
// double-precision MPFR oracles are valid references (compared in `double`,
// where any p-bit result is exact).

static double ref_add3(double x, double y, double z, int p, mpfx::RM rm) {
    mpfr_t mx, my, mz, acc, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mz, 53);
    mpfr_init2(acc, 256); // wide enough to sum three values exactly
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_set_d(mz, z, MPFR_RNDN);
    mpfr_add(acc, mx, my, MPFR_RNDN); // exact at 256 bits
    mpfr_add(acc, acc, mz, MPFR_RNDN); // exact at 256 bits
    mpfr_set(mr, acc, cvt_rm(rm)); // round the exact sum to p bits

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mz);
    mpfr_clear(acc);
    mpfr_clear(mr);
    return r;
}

static double ref_add4(double x, double y, double z, double w, int p, mpfx::RM rm) {
    mpfr_t mx, my, mz, mw, acc, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mz, 53);
    mpfr_init2(mw, 53);
    mpfr_init2(acc, 256); // wide enough to sum four values exactly
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_set_d(mz, z, MPFR_RNDN);
    mpfr_set_d(mw, w, MPFR_RNDN);
    mpfr_add(acc, mx, my, MPFR_RNDN); // exact at 256 bits
    mpfr_add(acc, acc, mz, MPFR_RNDN); // exact at 256 bits
    mpfr_add(acc, acc, mw, MPFR_RNDN); // exact at 256 bits
    mpfr_set(mr, acc, cvt_rm(rm)); // round the exact sum to p bits

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mz);
    mpfr_clear(mw);
    mpfr_clear(acc);
    mpfr_clear(mr);
    return r;
}

// rounding modes shared by the float tests
static const std::vector<mpfx::RM> F32_ROUNDING_MODES = {
    mpfx::RM::RNE,
    mpfx::RM::RTP,
    mpfx::RM::RTN,
    mpfx::RM::RTZ,
    mpfx::RM::RAZ,
};

TEST(OpsF32, TestAddEFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_add(x, y, p, rm);
                const float z = mpfx::add<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestSubEFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_sub(x, y, p, rm);
                const float z = mpfx::sub<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestMulEFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_mul(x, y, p, rm);
                const float z = mpfx::mul<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestDivEFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                if (y == 0.0f) continue; // skip division by zero

                const double z_ref = ref_div(x, y, p, rm);
                const float z = mpfx::div<mpfx::Engine::EFT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestSqrtEFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(0.0f, 4.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);

                const double z_ref = ref_sqrt(x, p, rm);
                const float z = mpfx::sqrt<mpfx::Engine::EFT>(x, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestFmaEFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                const float z = dist(rng);

                const double w_ref = ref_fma(x, y, z, p, rm);
                const float w = mpfx::fma<mpfx::Engine::EFT>(x, y, z, ctx);
                EXPECT_EQ(w_ref, static_cast<double>(w));
            }
        }
    }
}

TEST(OpsF32, TestAdd3EFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                const float z = dist(rng);

                const double w_ref = ref_add3(x, y, z, p, rm);
                const float w = mpfx::add3<mpfx::Engine::EFT>(x, y, z, ctx);
                EXPECT_EQ(w_ref, static_cast<double>(w));
            }
        }
    }
}

TEST(OpsF32, TestAdd4EFTUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                const float z = dist(rng);
                const float w = dist(rng);

                const double v_ref = ref_add4(x, y, z, w, p, rm);
                const float v = mpfx::add4<mpfx::Engine::EFT>(x, y, z, w, ctx);
                EXPECT_EQ(v_ref, static_cast<double>(v));
            }
        }
    }
}

// `float` variants using the FP_RTO engine (round-to-odd via FPU exceptions).
// Like EFT, it produces a correctly-rounded result for context precision
// p <= 22, so the same MPFR oracles apply.

TEST(OpsF32, TestAddFPUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_add(x, y, p, rm);
                const float z = mpfx::add<mpfx::Engine::FP_RTO>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestSubFPUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_sub(x, y, p, rm);
                const float z = mpfx::sub<mpfx::Engine::FP_RTO>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestMulFPUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_mul(x, y, p, rm);
                const float z = mpfx::mul<mpfx::Engine::FP_RTO>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestDivFPUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                if (y == 0.0f) continue; // skip division by zero

                const double z_ref = ref_div(x, y, p, rm);
                const float z = mpfx::div<mpfx::Engine::FP_RTO>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestSqrtFPUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(0.0f, 4.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);

                const double z_ref = ref_sqrt(x, p, rm);
                const float z = mpfx::sqrt<mpfx::Engine::FP_RTO>(x, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestFmaFPUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                const float z = dist(rng);

                const double w_ref = ref_fma(x, y, z, p, rm);
                const float w = mpfx::fma<mpfx::Engine::FP_RTO>(x, y, z, ctx);
                EXPECT_EQ(w_ref, static_cast<double>(w));
            }
        }
    }
}

// `float` variants using the SoftFloat engine (native round-to-odd mode).
// Correctly rounded for context precision p <= 22, so the MPFR oracles apply.

TEST(OpsF32, TestAddSFUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_add(x, y, p, rm);
                const float z = mpfx::add<mpfx::Engine::SOFTFLOAT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestSubSFUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_sub(x, y, p, rm);
                const float z = mpfx::sub<mpfx::Engine::SOFTFLOAT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestMulSFUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);

                const double z_ref = ref_mul(x, y, p, rm);
                const float z = mpfx::mul<mpfx::Engine::SOFTFLOAT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestDivSFUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                if (y == 0.0f) continue; // skip division by zero

                const double z_ref = ref_div(x, y, p, rm);
                const float z = mpfx::div<mpfx::Engine::SOFTFLOAT>(x, y, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestSqrtSFUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(0.0f, 4.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);

                const double z_ref = ref_sqrt(x, p, rm);
                const float z = mpfx::sqrt<mpfx::Engine::SOFTFLOAT>(x, ctx);
                EXPECT_EQ(z_ref, static_cast<double>(z));
            }
        }
    }
}

TEST(OpsF32, TestFmaSFUniform) {
    static constexpr size_t N = 200000;
    std::random_device r;
    std::mt19937_64 rng(r());
    for (int p = 2; p <= 8; p++) {
        for (const auto rm : F32_ROUNDING_MODES) {
            const mpfx::MPContext ctx(p, rm);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < N; i++) {
                const float x = dist(rng);
                const float y = dist(rng);
                const float z = dist(rng);

                const double w_ref = ref_fma(x, y, z, p, rm);
                const float w = mpfx::fma<mpfx::Engine::SOFTFLOAT>(x, y, z, ctx);
                EXPECT_EQ(w_ref, static_cast<double>(w));
            }
        }
    }
}
