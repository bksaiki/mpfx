// Diagnostic probe for the hardware round-to-odd path (`mpfx::engine_fp`).
//
// On arm64 the RTO engine returns the RTZ result without the sticky bit, which
// means `finalize()` never sees the inexact flag. This probe separates the two
// possible causes:
//
//   1. the flags are not raised/readable at all, or
//   2. the compiler moves the arithmetic across the `mrs`/`msr` inline asm,
//      so the flags are read before the operation executes.
//
// Build at both -O0 and -O2: if the naive path only breaks with optimization,
// or the barriered path fixes it, the cause is (2).

#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <mpfx/arch.hpp>
#include <mpfx/engine_eft.hpp>
#include <mpfx/engine_fp.hpp>

// Forces `v` to be materialized in an FP register here, ordering it against
// neighboring `asm volatile` statements.
template <typename T>
static inline void fp_barrier(T& v) {
#if defined(__aarch64__)
    __asm__ volatile("" : "+w"(v));
#elif defined(__x86_64__)
    __asm__ volatile("" : "+x"(v));
#else
    __asm__ volatile("" : "+g"(v));
#endif
}

// `engine_fp` with data dependencies pinning the operation between the two
// register accesses (the proposed fix).
template <typename Op>
static inline double rto_barriered(Op op, double x, double y) {
    const auto old = mpfx::arch::prepare_rto();
    fp_barrier(x);
    fp_barrier(y);
    double r = op(x, y);
    fp_barrier(r);
    const auto fexps = mpfx::arch::rto_status(old);
    if (fexps & mpfx::arch::EXCEPT_INEXACT) {
        uint64_t b;
        std::memcpy(&b, &r, sizeof(b));
        b |= 1;
        std::memcpy(&r, &b, sizeof(b));
    }
    return r;
}

static double add_barriered(double x, double y) {
    return rto_barriered([](double a, double b) { return a + b; }, x, y);
}

static uint64_t bits(double x) {
    uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

static int fails = 0;

static void check(const char* name, bool ok, const char* detail) {
    std::printf("  %-28s %s   %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) fails++;
}

int main() {
    // volatile so nothing is constant-folded away at compile time
    volatile double vx = 1.0;
    volatile double vy = 0x1p-60;

    std::printf("probe: arch=%s\n",
#if defined(MPFX_ARCH_X86)
        "x86"
#elif defined(MPFX_ARCH_ARM64)
        "arm64"
#else
        "generic"
#endif
    );

    // (1) does the C library see the inexact flag at all?
    {
        std::feclearexcept(FE_ALL_EXCEPT);
        volatile double s = vx + vy;
        (void) s;
        const bool ok = std::fetestexcept(FE_INEXACT) != 0;
        check("fetestexcept(INEXACT)", ok, ok ? "" : "libc does not observe inexact");
    }

    // (2) does the raw arch layer see it?
    {
        mpfx::arch::clear_exceptions();
        volatile double s = vx + vy;
        (void) s;
        const auto e = mpfx::arch::get_exceptions();
        const bool ok = (e & mpfx::arch::EXCEPT_INEXACT) != 0;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "flags=0x%x", (unsigned) e);
        check("arch::get_exceptions", ok, buf);
    }

    // (3) does the rounding mode round-trip?
    {
        const int before = mpfx::arch::get_rounding_mode();
        mpfx::arch::set_rounding_mode(mpfx::arch::RM_RTZ);
        const int got = mpfx::arch::get_rounding_mode();
        mpfx::arch::set_rounding_mode(before);
        const bool ok = got == mpfx::arch::RM_RTZ;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "set RTZ(%d) -> %d", mpfx::arch::RM_RTZ, got);
        check("rounding mode round-trip", ok, buf);
    }

    // (4) the RTO engine itself. 1 + 2^-60 truncates to 1.0 (even LSB), so the
    // sticky bit must turn it into 0x3ff0000000000001.
    for (int neg = 0; neg < 2; neg++) {
        const double x = neg ? -vx : vx;
        const double y = neg ? -vy : vy;
        const uint64_t want = bits(x) | 1;

        const double got = mpfx::engine_fp::add(x, y, 53);
        const double got_b = add_barriered(x, y);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "%016llx (want %016llx)",
                      (unsigned long long) bits(got), (unsigned long long) want);
        check(neg ? "engine_fp::add negative" : "engine_fp::add positive",
              bits(got) == want, buf);

        std::snprintf(buf, sizeof(buf), "%016llx (want %016llx)",
                      (unsigned long long) bits(got_b), (unsigned long long) want);
        check(neg ? "  barriered negative" : "  barriered positive",
              bits(got_b) == want, buf);
    }

    // (5) Replicate the failing call site: the same loop shape as
    // tests/test_engine.cpp, where the operands arrive from the RNG. Comparing
    // plain vs. barriered `engine_fp` is self-contained -- if they disagree, the
    // operation is being scheduled outside the RTO window.
    {
        constexpr size_t N = 500000;
        std::mt19937_64 rng(20260813);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::pair<double, double>> in;
        in.reserve(N);
        for (size_t i = 0; i < N; i++) in.emplace_back(dist(rng), dist(rng));

        static const struct {
            const char* name;
            double (*fp)(double, double, mpfx::prec_t);
            double (*eft)(double, double, mpfx::prec_t);
            double (*bar)(double, double);
        } ops[] = {
            {"add", mpfx::engine_fp::add, mpfx::engine_eft::add,
             [](double a, double b) { return rto_barriered([](double p, double q) { return p + q; }, a, b); }},
            {"sub", mpfx::engine_fp::sub, mpfx::engine_eft::sub,
             [](double a, double b) { return rto_barriered([](double p, double q) { return p - q; }, a, b); }},
            {"mul", mpfx::engine_fp::mul, mpfx::engine_eft::mul,
             [](double a, double b) { return rto_barriered([](double p, double q) { return p * q; }, a, b); }},
            {"div", mpfx::engine_fp::div, mpfx::engine_eft::div,
             [](double a, double b) { return rto_barriered([](double p, double q) { return p / q; }, a, b); }},
        };

        std::printf("\n  loop over %zu random pairs (mismatches):\n", N);
        std::printf("  %-5s %12s %12s %12s\n", "op", "fp-vs-eft", "fp-vs-barr", "barr-vs-eft");
        for (const auto& op : ops) {
            size_t a = 0, b = 0, c = 0;
            for (const auto& [x, y] : in) {
                // same shape as the test: fp and eft back to back
                const double r_fp = op.fp(x, y, 53);
                const double r_eft = op.eft(x, y, 53);
                const double r_bar = op.bar(x, y);
                if (bits(r_fp) != bits(r_eft)) a++;
                if (bits(r_fp) != bits(r_bar)) b++;
                if (bits(r_bar) != bits(r_eft)) c++;
            }
            std::printf("  %-5s %12zu %12zu %12zu\n", op.name, a, b, c);
            if (a || b || c) fails++;
        }
    }

    std::printf("probe: %d check(s) failed\n", fails);
    return 0; // diagnostic only; never fail the job
}
