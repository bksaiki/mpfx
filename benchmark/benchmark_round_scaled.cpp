/// @file benchmark_round_scaled.cpp
/// @brief Compares the scale-and-truncate rounding implementations against the
/// existing integer-domain one.
///
/// Both implementations are measured twice, once computing no status flags and once
/// computing all of them, since everything reachable from `Context` asks for
/// `ALL_FLAGS` and the two implementations do not pay the same price for them:
///
///   base_nf, base_af      `experimental::round`, the integer-domain implementation
///   scaled_nf, scaled_af  `round_scaled`, the scaled implementation
///
/// The `nf`/`af` pairs are what to compare - a flag-computing implementation against
/// a flag-computing one. The relative table reports each against the matching
/// baseline for that reason.
///
/// The rounding mode is a compile-time parameter here, so the numbers isolate the
/// cost of rounding rather than of the runtime mode dispatch that `Context::round`
/// goes through. Timing follows the other benchmarks in this directory: a
/// `volatile` sink per iteration, which serializes the loop and therefore reports
/// something closer to latency than to throughput. That is the pessimistic reading
/// and the one comparable to `benchmark_round.cpp`.
///
/// The working set is sized to stay in cache and is swept repeatedly, rather than
/// allocating one enormous vector, so the measurement reflects computation rather
/// than memory bandwidth. Both the minimum and the median across repetitions are
/// reported, because these timings are noisy and a single run is not trustworthy.
///
/// Usage: benchmark_round_scaled [num_inputs] [reps]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "mpfx/round.hpp"

namespace {

using clock_t_ = std::chrono::steady_clock;
using mpfx::bit_float;
using mpfx::exp_t;
using mpfx::prec_t;
using mpfx::RM;

/// @brief Which implementation to measure.
enum class Impl { BASE_NF, BASE_AF, SCALED_NF, SCALED_AF };

constexpr Impl ALL_IMPLS[] = {Impl::BASE_NF, Impl::BASE_AF, Impl::SCALED_NF, Impl::SCALED_AF};
constexpr RM ALL_MODES[] = {
    RM::RNE, RM::RNA, RM::RTP, RM::RTN, RM::RTZ, RM::RAZ, RM::RTO, RM::RTE,
};

const char* impl_name(Impl impl) {
    switch (impl) {
    case Impl::BASE_NF: return "base_nf";
    case Impl::BASE_AF: return "base_af";
    case Impl::SCALED_NF: return "scaled_nf";
    case Impl::SCALED_AF: return "scaled_af";
    }
    return "?";
}

const char* rm_name(RM rm) {
    switch (rm) {
    case RM::RNE: return "rne";
    case RM::RNA: return "rna";
    case RM::RTP: return "rtp";
    case RM::RTN: return "rtn";
    case RM::RTZ: return "rtz";
    case RM::RAZ: return "raz";
    case RM::RTO: return "rto";
    case RM::RTE: return "rte";
    }
    return "?";
}

/// @brief Applies one implementation, selected at compile time.
template <Impl impl, RM rm, std::floating_point T>
inline bit_float<T> apply(bit_float<T> x, prec_t p, std::optional<exp_t> n) {
    using namespace mpfx;
    if constexpr (impl == Impl::BASE_NF) {
        return experimental::round<rm, Flags::NO_FLAGS>(x, p, n);
    } else if constexpr (impl == Impl::BASE_AF) {
        return experimental::round<rm, Flags::ALL_FLAGS>(x, p, n);
    } else if constexpr (impl == Impl::SCALED_NF) {
        return experimental::round_scaled<rm, Flags::NO_FLAGS>(x, p, n);
    } else {
        return experimental::round_scaled<rm, Flags::ALL_FLAGS>(x, p, n);
    }
}

/// @brief Times a single pass over the inputs, in nanoseconds per operation.
template <Impl impl, RM rm, std::floating_point T>
double time_pass(const std::vector<T>& xs, prec_t p, std::optional<exp_t> n) {
    volatile T sink = static_cast<T>(0);
    const auto start = clock_t_::now();
    for (const T x : xs) {
        sink = apply<impl, rm>(bit_float<T>(x), p, n).to_float();
    }
    const auto end = clock_t_::now();
    (void) sink;
    const auto ns = std::chrono::duration<double, std::nano>(end - start).count();
    return ns / static_cast<double>(xs.size());
}

/// @brief Nanoseconds per operation across repetitions.
struct Timing {
    double min;
    double median;
    double spread; // max/min, as a check on how trustworthy the row is
};

/// @brief Summarizes one implementation's passes.
Timing summarize(std::vector<double> per_op) {
    std::sort(per_op.begin(), per_op.end());
    return {per_op.front(), per_op[per_op.size() / 2], per_op.back() / per_op.front()};
}

/// @brief One input distribution together with the format it rounds to.
template <std::floating_point T>
struct Workload {
    const char* name;
    prec_t p;
    std::optional<exp_t> n;
    std::vector<T> xs;
};

/// @brief Builds the input distributions.
///
/// `prec` and `emul` are the ordinary cases: values of moderate magnitude rounded
/// to a middling precision, without and with subnormalization. `narrow` emulates
/// an FP8-like format, whose smallest positive value is large enough that ordinary
/// inputs fall below it, so it exercises the underflow-to-zero path heavily.
/// `bits` draws whole encodings uniformly, which reaches subnormals and the far
/// ends of the exponent range that the other three never touch.
template <std::floating_point T>
std::vector<Workload<T>> workloads(size_t n_inputs) {
    using params_t = typename bit_float<T>::params_t;
    static constexpr prec_t HALF_P = params_t::P / 2;

    std::mt19937_64 rng(0xB0FFA);
    std::uniform_real_distribution<double> mant(1.0, 2.0);
    std::bernoulli_distribution sign(0.5);

    // moderate magnitudes, spanning a few binades
    const auto reals = [&](int lo, int hi) {
        std::uniform_int_distribution<int> e(lo, hi);
        std::vector<T> v(n_inputs);
        for (auto& x : v) {
            const double y = std::ldexp(mant(rng), e(rng));
            x = static_cast<T>(sign(rng) ? -y : y);
        }
        return v;
    };

    // whole encodings, skipping NaN and infinity: both implementations return
    // those immediately, so including them would only dilute the measurement
    const auto encodings = [&]() {
        using uint_t = typename bit_float<T>::uint_t;
        std::uniform_int_distribution<uint_t> bits(0, ~static_cast<uint_t>(0));
        std::vector<T> v(n_inputs);
        for (auto& x : v) {
            bit_float<T> b(static_cast<uint_t>(bits(rng)));
            while (b.is_nar()) {
                b = bit_float<T>(static_cast<uint_t>(bits(rng)));
            }
            x = b.to_float();
        }
        return v;
    };

    std::vector<Workload<T>> out;
    out.push_back({"prec", HALF_P, std::nullopt, reals(-20, 5)});
    out.push_back({"emul", HALF_P, params_t::EMIN - static_cast<exp_t>(HALF_P), reals(-20, 5)});
    out.push_back({"narrow", 4, -10, reals(-30, 5)});
    out.push_back({"bits", HALF_P, params_t::EXPMIN - 1, encodings()});
    return out;
}

/// @brief Measures every implementation for one workload and one mode, and prints
/// a row of the raw table plus a row of the relative table.
///
/// The implementations are interleaved within each repetition rather than each
/// being run to completion in turn. On a machine whose clock drifts as it warms up
/// - a laptop, say - measuring them one after another would systematically favour
/// whichever went first, and that bias would be invisible in the output.
template <RM rm, std::floating_point T>
void run_row(const char* container, const Workload<T>& w, size_t reps,
             std::string& raw, std::string& rel) {
    static constexpr size_t N_IMPLS = std::size(ALL_IMPLS);
    std::vector<double> passes[N_IMPLS];

    for (size_t r = 0; r < reps; r++) {
        passes[0].push_back(time_pass<Impl::BASE_NF, rm>(w.xs, w.p, w.n));
        passes[1].push_back(time_pass<Impl::BASE_AF, rm>(w.xs, w.p, w.n));
        passes[2].push_back(time_pass<Impl::SCALED_NF, rm>(w.xs, w.p, w.n));
        passes[3].push_back(time_pass<Impl::SCALED_AF, rm>(w.xs, w.p, w.n));
    }

    Timing t[N_IMPLS];
    for (size_t i = 0; i < N_IMPLS; i++) {
        t[i] = summarize(passes[i]);
    }

    char line[512];
    int k = snprintf(line, sizeof(line), "%s,%s,%s", container, w.name, rm_name(rm));
    for (const auto& x : t) {
        k += snprintf(line + k, sizeof(line) - k, ",%.3f,%.3f,%.2f", x.min, x.median, x.spread);
    }
    raw += line;
    raw += "\n";

    // speedup over the existing implementation at the same flag mask, on the minimums
    snprintf(line, sizeof(line), "%s,%s,%s,%.2f,%.2f", container, w.name, rm_name(rm),
             t[0].min / t[2].min, t[1].min / t[3].min);
    rel += line;
    rel += "\n";
}

/// @brief Sweeps every workload and every rounding mode for one container type.
template <std::floating_point T>
void run_container(const char* container, size_t n_inputs, size_t reps,
                   std::string& raw, std::string& rel) {
    for (const auto& w : workloads<T>(n_inputs)) {
        // unroll over the rounding modes so that each stays a compile-time constant
        [&]<size_t... I>(std::index_sequence<I...>) {
            (run_row<ALL_MODES[I]>(container, w, reps, raw, rel), ...);
        }(std::make_index_sequence<std::size(ALL_MODES)>{});
    }
}

} // namespace

int main(int argc, char** argv) {
    const size_t n_inputs = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : (1u << 20);
    const size_t reps = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 5;
    if (n_inputs == 0 || reps == 0) {
        std::fprintf(stderr, "usage: %s [num_inputs] [reps]\n", argv[0]);
        return 1;
    }

    std::string raw, rel;
    run_container<float>("f32", n_inputs, reps, raw, rel);
    run_container<double>("f64", n_inputs, reps, raw, rel);

    std::printf("# ns per operation (%zu inputs, %zu reps). `spread` is max/min across\n"
                "# repetitions: treat a row whose spread is far above 1 with suspicion, and\n"
                "# do not read anything into differences smaller than it.\n",
                n_inputs, reps);
    std::printf("container,workload,mode");
    for (const auto impl : ALL_IMPLS) {
        std::printf(",%s_min,%s_med,%s_spread", impl_name(impl), impl_name(impl), impl_name(impl));
    }
    std::printf("\n%s\n", raw.c_str());

    std::printf("# speedup of scaled over base at the same flag mask, on the minimums\n");
    std::printf("container,workload,mode,no_flags,all_flags\n%s\n", rel.c_str());
    return 0;
}
