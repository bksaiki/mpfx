# Roadmap

Open threads for the library, forward-looking only; drop an item when
it lands. References are headers under `include/mpfx/`, the suites
under `tests/`, and the programs under `examples/`, `benchmark/`, and
`scripts/`, never line numbers.

## Correctness

- **Pad per mode and validate the target format.**
  `Context::round_prec` pads every mode by two digits, and the value
  reaches only debug assertions: each round-to-odd engine (hardware,
  EFT, SoftFloat, FloppyFloat) works at the container's full precision
  regardless. So the pad can follow the paper's double-rounding rules
  at no cost: none for RTO, one digit for the directed modes, two for
  the nearest modes. Then check containment of the padded format in
  the container rather than its precision alone, which also admits
  fixed-point targets such as `Context(200, −13, 100)` for
  A(∞, −12, 100): the precision assertion rejects them although
  rounding handles their values (the tininess flags fire on every
  result, since the whole format sits below its normal range).
  Decide what a release build does outside the class. Today it
  accepts p = 52 and 53 for `double` and answers wrongly (the nearest
  modes at 52, every mode but RTO at 53: `add(1, 2^−52 + 2^−56)`
  under RNE at 52, `add(1, 2^−53 + 2^−56)` under RTZ at 53), accepts
  exp = −1073 and answers wrongly under RNE
  (`mul(1.375 × 2^−1017, 2^−56)` into A(8, −1073, 1)), and no build
  checks the padded exponent or the bound. The class to state per
  mode is the rules' premise with the container as the intermediate
  format, A(53, −1074, `DBL_MAX`) or A(24, −149, `FLT_MAX`).
  Its precision and exponent parts, in `double`:
  p ≤ 52 with exp ≥ −1073 for the directed modes,
  p ≤ 53 with exp ≥ −1074 for RTO,
  p ≤ 51 with exp ≥ −1072 for the nearest modes;
  in `float`, p ≤ 23, 24, and 22 with exp ≥ −148, −149, and −147.
- **The EFT engine loses small error terms.** In `mul`, `div`,
  `sqrt`, and `fma` the error term is computed in binary64, so one
  below 2^−1074 can round to zero, and `engine_eft::round_finalize`
  then returns the nearest result instead of round-to-odd. So some
  small results inside the class come out wrong: `mul(x, x)` with
  x = (1 + 2^−52) 2^−510 into A(51, −1072) under RNE rounds down
  where the exact result rounds up, and the hardware engine is right.
  `add` and `sub` are unaffected, since TwoSum stays exact. Scale the
  operands into range, or narrow the EFT engine's class.
- **RTE.** `RoundingMode::RTE` is an eighth mode; the paper's rules
  cover seven. One digit of round-to-odd padding on precision and
  exponent together serves it on every small format swept (the Lean
  development's roadmap names the set), and nothing is proved.
  Support it with a rule and padding of its own, document it as
  outside the guarantee, or drop the mode, which no context or
  example uses.
- **Two engine values the paper does not describe.** `Engine::FIXED`
  is implemented for `mul` only, used nowhere, and its `int64_t`
  product can overflow; `Engine::FP_EXACT` re-rounds the hardware's
  nearest result, so it relies on the operation being exact, and
  appears in one example. The paper describes four backends. Prune
  or document.
- **Overflow edges.**
  - `round_overflow` returns early on a non-finite value, so when the
    final rounding carries to 2^1024 (`round(DBL_MAX)` at p = 4, RNE)
    no overflow is flagged, and a `Context` built with
    `OverflowMode::TO_MAXVAL` returns infinity; `EFloatContext`, the
    only context that selects that mode, saturates in its `fixup`.
  - `EFloatContext` sends overflow to NaN for a format with NaN but
    no infinity (E4M3 under RNE), where the paper says formats
    without infinities saturate to ±b. Pick one.
  - On overflow the final rounding sends an RTO target to ±∞
    (`overflow_to_infinity`), while the round-to-odd intermediate,
    which never rounds to infinity, saturates to ±`DBL_MAX` when the
    container overflows. Pick one, and pin it in the exhaustive
    suite's oracle.
  - In debug builds the hardware engine asserts that the operation
    raised neither overflow nor underflow, so a tiny inexact
    intermediate aborts even for a target the rules license
    (`mul(1.375 × 2^−1017, 2^−56)` into A(8, −1072, 1)), and so does
    any overflowing one that the final rounding would have handled.
  - From `DBL_MAX` plus half an ulp upward the EFT engine's nearest
    result is ±∞ although the round-to-odd result is finite: debug
    builds abort in `engine_eft::round_finalize`, and release builds
    return NaN for fma, add3, and add4 (their error terms compute
    inf − inf; on arm64, x86 not tried), so
    `fma(DBL_MAX, 1, 2^970)` into bfloat16 under RTZ gives NaN, not
    the bfloat16 maximum, and the right answer for add, sub, mul,
    and div only because the infinity encoding minus one is
    `DBL_MAX`.

## Testing (`tests/`)

- **Extend `test_ops`.** It draws 1,000,000 uniform doubles
  on [−1, 1] ([0, 1] for sqrt) for each operation, mode, and
  precision from 2 to 8, and compares RNE, RTP, RTN, RTZ, and RAZ
  against MPFR at the target precision; 200,000 for `float`
  ([0, 4] for sqrt). Its targets are `MPContext`, A(p, −∞, ∞),
  outside the class the library is correct on; its inputs never
  reach the container's quantum, where such targets fail. After the
  exhaustive suite lands, add what neither covers: wider precisions
  and bounds inside the class, with all four engines and seven modes
  on `double`; results near the bottom and top of the container's
  range; and `add3` and `add4` on `double`.
- **Land the exhaustive suite.** The 14 IEEE-style formats of at most
  eight bits with a 2- to 5-bit exponent field (subnormals, signed
  zeros, infinities), six operations, seven modes, the hardware,
  EFT, SoftFloat, and FloppyFloat engines, both containers: 4,704
  configurations, MPFR oracle bit-for-bit including the sign of zero.
  It lives outside this repository; bring it in as a script or a
  slow test so it can be rerun, noting the commit it last passed on.
- **`TestOverflowFlag` cannot reach the lost flag.** It rounds exact
  values of at most p bits with exponents from −4 to 4, where its
  oracle, |x| > bound, is right. Reach the carry to 2^1024 under
  "Overflow edges"; for inexact inputs, compare the bound with the
  value rounded as if unbounded, not with x (at p = 4, RNE rounds
  10.25 to 10, which does not overflow a bound of 10).

## Examples and benchmarks

- `examples/mx_dot_prod.cpp` never checks that its SoftFloat and MPFX
  results agree, and two defects stand in the way. The quantizer's
  `Context` does not saturate, so an element that rounds above the
  format's maximum becomes an infinity (1.9 × 2^15 at p = 3 rounds
  to 2^16 > 57344), and `to_fixed` returns `int64_t`, which an
  E5M2 × E5M2 product at quantum 2^−32 overflows (57344² needs
  49 × 2^58). `mx_dot_prod_ref`, which it never calls, adds the block
  scales where the other two multiply them. Fix these, then check
  SoftFloat against MPFX bit for bit, so the case study is tested,
  not only timed.
- `examples/mixed_dot_prod.cpp` accumulates in
  `IEEE754Context(11, 32)`, commented as FP32, which is (8, 32).
- The paper's performance comparison comes from a harness outside
  the tree: `benchmark/benchmark_ops.cpp` times FP32 targets only,
  once per run, without CPFloat, and `scripts/benchmark.py` runs a
  `build/benchmark/ops` the build does not produce and passes it no
  arguments, while the paper has FP16 targets and CPFloat per element
  and on arrays of 10M inputs. Bring the harness in, with the commit
  it ran on, so the comparison can be rerun.
- A SoftFloat implementation with the same transformations as the
  MPFX one in `mx_dot_prod`, so the reported speedup isolates the
  library's contribution.

## Later

- A batched interface that hoists the per-format setup (subnormal
  thresholds, overflow bound, significand masks) out of the
  per-operation path.
- A same-mode engine for directed targets, which the rules license
  with no padding.
