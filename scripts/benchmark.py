import csv
import math
import pickle
import subprocess

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.resolve()
BUILD_DIR = REPO_ROOT / "build"

# Target formats and rounding modes to sweep over. The values are the CLI
# tokens understood by `benchmark_ops`. Every (format, rounding mode) pair is
# benchmarked, reported, and plotted independently.
FP32, FP16 = 'fp32', 'fp16'
FORMATS = [FP32, FP16]

RNE, RTP, RTZ = 'rne', 'rtp', 'rtz'
RMS = [RNE, RTP, RTZ]

ROWS = [
    'add',
    'sub',
    'mul',
    'div',
    'sqrt',
    'fma',
]

# Columns emitted by `benchmark_ops` (the native-hardware column was removed:
# there is no portable scalar half-precision type to time against).
COLUMNS = [
    'mpfr',
    'softfloat',
    'floppyfloat',
    'mpfx_rto',
    # 'mpfx_sfloat',
    # 'mpfx_ffloat',
    'mpfx_eft'
]

# Baseline that overheads/speedups are measured against. `benchmark_ops` reports
# its own speedup table relative to SoftFloat, so we match that here.
BASELINE = 'softfloat'

# Columns shown in tables/plots (everything except the baseline itself).
DISPLAY_COLUMNS = [c for c in COLUMNS if c != BASELINE]

NAMES = {
    'mpfr': 'MPFR',
    'softfloat': 'SoftFloat',
    'floppyfloat': 'FloppyFloat',
    'mpfx_rto': 'MPFX (RTO)',
    'mpfx_sfloat': 'MPFX (SoftFloat)',
    'mpfx_ffloat': 'MPFX (FloppyFloat)',
    'mpfx_eft': 'MPFX (EFT)',
}


def parse_time(s: str) -> float:
    # `benchmark_ops` prints `n/a` for columns that were not run (e.g. the
    # FloppyFloat column under FP16, which has no f16 instantiation).
    return math.nan if s.strip() == 'n/a' else float(s)


def nan_mean(values: list[float]) -> float:
    present = [v for v in values if not math.isnan(v)]
    return sum(present) / len(present) if present else math.nan


@dataclass
class TaskConfig:
    task_id: int       # globally unique id across the whole sweep
    iteration: int     # iteration index within this (format, rounding mode)
    cache_dir: Path
    num_inputs: int
    rounding_mode: str
    fmt: str


def benchmark_task(config: TaskConfig):
    # run benchmark and capture output to parse as CSV
    print(f"Running benchmark task {config.task_id} (iter {config.iteration}, {config.fmt}, {config.rounding_mode})...")
    benchmark_path = BUILD_DIR / "benchmark" / "benchmark_ops"
    p = subprocess.run(
        [str(benchmark_path), str(config.num_inputs), config.rounding_mode, config.fmt],
        capture_output=True,
        check=True,
    )
    output = p.stdout.decode()

    # `benchmark_ops` prints a `type: <fmt>` preamble line, then two tables: raw
    # runtimes, a blank line, a `# speedup ...` comment, and a speedup table.
    # Skip the preamble, lock onto the `op,...` header, and parse only the first
    # (runtime) table -- stop at the first blank line or comment.
    data: list[list[str]] = []
    in_table = False
    for line in output.splitlines():
        stripped = line.strip()
        if not in_table:
            # find the runtime table's header row
            if stripped.split(',', 1)[0].strip() == 'op':
                in_table = True
                data.append([datum.strip() for datum in next(csv.reader([line]))])
            continue
        if not stripped or stripped.startswith('#'):
            break
        data.append([datum.strip() for datum in next(csv.reader([line]))])

    # write raw output to cache
    cache_file = config.cache_dir / f"raw_{config.fmt}_{config.rounding_mode}_iter_{config.iteration}.csv"
    with cache_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Completed benchmark task {config.task_id} (iter {config.iteration}, {config.fmt}, {config.rounding_mode}).")
    return config.fmt, config.rounding_mode, data


def run_all(output_dir: Path, iterations: int, threads: int, num_inputs: int):
    cache_dir = output_dir / "cache"
    plot_dir = output_dir / "plots"

    # Ensure directories exists
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # build one flat task list across every (format, rounding mode, iteration)
    # so all of them share a single process pool. `task_id` is unique across the
    # whole sweep; `iteration` is the per-(format, rounding mode) repeat index.
    configs: list[TaskConfig] = []
    task_id = 0
    for fmt in FORMATS:
        for rm in RMS:
            for i in range(iterations):
                configs.append(TaskConfig(task_id, i, cache_dir, num_inputs, rm, fmt))
                task_id += 1

    # run every task in parallel, regrouping results by (format, rounding mode)
    print(f'Running {len(configs)} configurations...')
    grouped: dict[tuple[str, str], list] = {(fmt, rm): [] for fmt in FORMATS for rm in RMS}
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(benchmark_task, config) for config in configs]
        for future in futures:
            fmt, rm, data = future.result()
            grouped[(fmt, rm)].append(data)

    print("All benchmark tasks complete; aggregating and caching...")

    # aggregate and cache each (format, rounding mode) independently
    for (fmt, rm), results in grouped.items():
        aggregate_and_cache(cache_dir, results, fmt, rm)


def aggregate_and_cache(cache_dir: Path, results: list, fmt: str, rounding_mode: str):
    # group first by operation, then by header1..N
    # discard the header row
    aggregated: dict[str, dict[str, list[float]]] = {}
    for result in results:
        headers = result[0]
        # COLUMNS is a (possibly pruned) selection of what the binary emits, so
        # require it to be a subset rather than an exact match. The baseline must
        # always be present since overheads are computed against it. Every emitted
        # column is still cached, so pruning only affects what gets displayed.
        emitted = headers[1:]
        missing = [c for c in COLUMNS if c not in emitted]
        assert not missing, f"benchmark output is missing requested columns {missing} (emitted: {emitted})"
        assert BASELINE in emitted, f"baseline '{BASELINE}' missing from benchmark output: {emitted}"
        for row in result[1:]:
            op_name = row[0]
            if op_name not in aggregated:
                table: dict[str, list[float]] = {}
                for header, time in zip(headers[1:], row[1:]):
                    table[header] = [parse_time(time)]
                aggregated[op_name] = table
            else:
                for header, time in zip(headers[1:], row[1:]):
                    aggregated[op_name][header].append(parse_time(time))

    print(f"Aggregated benchmark results for {fmt}/{rounding_mode}.")

    # compute average time
    average_runtimes: dict[tuple[str, str], float] = {}
    for op_name, table in aggregated.items():
        for header, times in table.items():
            average_runtimes[(op_name, header)] = nan_mean(times)

    # write average runtimes to pickle
    avg_runtime_file = cache_dir / f"average_runtimes_{fmt}_{rounding_mode}.pkl"
    with avg_runtime_file.open('wb') as f:
        pickle.dump(average_runtimes, f)

    # compute average overhead relative to the baseline (e.g. SoftFloat)
    average_overheads: dict[tuple[str, str], float] = {}
    for op_name, table in aggregated.items():
        baseline = nan_mean(table[BASELINE])
        for header, times in table.items():
            if header != BASELINE:
                average_overheads[(op_name, header)] = nan_mean(times) / baseline

    # write average overheads to pickle
    avg_overhead_file = cache_dir / f"average_overheads_{fmt}_{rounding_mode}.pkl"
    with avg_overhead_file.open('wb') as f:
        pickle.dump(average_overheads, f)


def report_overhead(output_dir: Path, fmt: str, rm: str):
    # load average overheads from pickle
    avg_overhead_file = output_dir / "cache" / f"average_overheads_{fmt}_{rm}.pkl"
    with avg_overhead_file.open('rb') as f:
        average_overheads: dict[tuple[str, str], float] = pickle.load(f)

    print(f"\n# {fmt}/{rm}: runtime relative to {NAMES[BASELINE]} (>1 = slower, <1 = faster)")
    print(f'{"op":<12}', end="")
    for col in DISPLAY_COLUMNS:
        print(f"{col:>12}", end="")
    print()

    for row in ROWS:
        print(f"{row:<12}", end="")
        for col in DISPLAY_COLUMNS:
            overhead = average_overheads[(row, col)]
            cell = "n/a" if math.isnan(overhead) else f"{overhead:.2f}"
            print(f'{cell:>12}', end="")
        print()


def _rm_shade(base, r: int):
    # Lightness encodes rounding mode: early modes lighter, last mode full color.
    frac = 0.45 + 0.55 * (r / max(len(RMS) - 1, 1))
    c = np.array(base) * frac + np.array([1.0, 1.0, 1.0, 1.0]) * (1.0 - frac)
    c[3] = 1.0
    return c


def plot_speedup(output_dir: Path, bare: bool = False):
    # One merged figure: a subplot row per format; within each subplot the bars
    # are grouped by operation, then clustered by treatment (hue), then by
    # rounding mode (shade). Heights are speedup relative to SoftFloat (>1 =
    # faster), i.e. the reciprocal of the cached overhead.
    cache_dir = output_dir / "cache"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # speedups[(fmt, rm)][(op, col)] = SoftFloat_time / impl_time
    speedups: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for fmt in FORMATS:
        for rm in RMS:
            with (cache_dir / f"average_overheads_{fmt}_{rm}.pkl").open('rb') as f:
                overheads: dict[tuple[str, str], float] = pickle.load(f)
            speedups[(fmt, rm)] = {
                key: (1.0 / ov if (ov and not math.isnan(ov)) else math.nan)
                for key, ov in overheads.items()
            }

    treatments = DISPLAY_COLUMNS  # SoftFloat is the 1.0 reference, not a bar
    R = len(RMS)
    T = len(treatments)

    # bar geometry (data units): all bars within one operator are flush (no gaps
    # between treatments or rounding modes); only operators are separated
    bar_w = 1.0
    treat_gap = 0.0
    op_gap = 2.0

    # shared x layout: (x, op, treatment, rm_index) and per-op group centers
    layout: list[tuple[float, str, str, int]] = []
    op_centers: list[float] = []
    x = 0.0
    for op in ROWS:
        start = x
        for col in treatments:
            for r in range(R):
                layout.append((x, op, col, r))
                x += bar_w
            x += treat_gap
        x -= treat_gap
        op_centers.append((start + x) / 2.0)
        x += op_gap
    extent = x - op_gap

    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:T]
    col_index = {c: i for i, c in enumerate(treatments)}

    fig, axes = plt.subplots(
        len(FORMATS), 1,
        figsize=(max(14.0, extent * 0.16), 3 * len(FORMATS) + 0.2),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, fmt in zip(axes, FORMATS):
        for (xpos, op, col, r) in layout:
            height = speedups[(fmt, RMS[r])].get((op, col), math.nan)
            if math.isnan(height):
                continue  # e.g. FloppyFloat under FP16 (not run)
            ax.bar(xpos, height, width=bar_w,
                   color=_rm_shade(base_colors[col_index[col]], r),
                   edgecolor='black', linewidth=0.3)

        ax.axhline(1.0, color='red', linewidth=0.9, linestyle='--', alpha=0.7)
        ax.set_xticks(op_centers)
        ax.set_xticklabels([op.upper() for op in ROWS])
        ax.set_xlim(-bar_w, extent)
        ax.set_ylabel(f'Speedup vs. {NAMES[BASELINE]}')
        ax.set_title(fmt.upper(), fontsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # two legends: hue distinguishes treatment, shade distinguishes rounding mode
    treat_handles = [plt.Rectangle((0, 0), 1, 1, fc=_rm_shade(base_colors[i], R - 1),
                                   edgecolor='black', linewidth=0.5) for i in range(T)]
    treat_labels = [NAMES[c] for c in treatments]
    rm_handles = [plt.Rectangle((0, 0), 1, 1, fc=_rm_shade(np.array([0.35, 0.35, 0.35, 1.0]), r),
                                edgecolor='black', linewidth=0.5) for r in range(R)]
    rm_labels = [rm.upper() for rm in RMS]

    # `bare` mode strips the title and legend for figure inclusion. Otherwise
    # lay out with the suptitle hugging the top row and the legends just beneath;
    # bbox_inches='tight' crops the rest either way.
    if bare:
        plt.tight_layout(rect=[0.01, 0.0, 1, 0.98])
    else:
        plt.suptitle(f'Speedup Relative to {NAMES[BASELINE]}', fontsize=16, y=0.965)
        plt.tight_layout(rect=[0.01, 0.0, 1, 0.93])

        leg1 = fig.legend(treat_handles, treat_labels, loc='upper center',
                          bbox_to_anchor=(0.5, -0.005), ncol=T, frameon=True,
                          fontsize=10, title='Treatment (color)')
        fig.add_artist(leg1)
        fig.legend(rm_handles, rm_labels, loc='upper center',
                   bbox_to_anchor=(0.5, -0.055), ncol=R, frameon=True,
                   fontsize=10, title='Rounding mode (shade: light→dark)')

    plot_file = plot_dir / "speedup.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {plot_file}")



def report_tex(output_dir: Path):
    # Emit LaTeX table rows of speedup relative to SoftFloat (>1 = faster).
    # One row per (op, format); each cell holds the rounding modes in RMS order,
    # slash-separated. Columns follow COLUMNS (SoftFloat is 1.00 by definition).
    cache_dir = output_dir / "cache"

    # load per-(format, rounding mode) average runtimes
    runtimes: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for fmt in FORMATS:
        for rm in RMS:
            with (cache_dir / f"average_runtimes_{fmt}_{rm}.pkl").open('rb') as f:
                runtimes[(fmt, rm)] = pickle.load(f)

    lines: list[str] = []
    for fmt_idx, fmt in enumerate(FORMATS):
        for op in ROWS:
            cells: list[str] = []
            for col in COLUMNS:
                vals: list[str] = []
                for rm in RMS:
                    rt = runtimes[(fmt, rm)]
                    base = rt[(op, BASELINE)]
                    t = rt[(op, col)]
                    # speedup = SoftFloat time / this implementation's time
                    speedup = base / t if t and not math.isnan(t) else math.nan
                    vals.append("n/a" if math.isnan(speedup) else f"{speedup:.2f}")
                cells.append("/".join(vals))
            lines.append(f"{op} ({fmt.upper()}) & " + " & ".join(cells) + r" \\")
        if fmt_idx != len(FORMATS) - 1:
            lines.append(r"\hline")

    table = "\n".join(lines)
    rms_label = "/".join(rm.upper() for rm in RMS)
    print(f"\n% LaTeX rows: speedup vs {NAMES[BASELINE]}, cells are {rms_label}")
    print(f"% columns: Op & " + " & ".join(NAMES[c] for c in COLUMNS) + r" \\")
    print(table)

    tex_file = output_dir / "overhead_table.tex"
    tex_file.write_text(table + "\n")
    print(f"\nSaved TeX rows: {tex_file}")


def build_benchmarks():
    # Navigate to build directory and build benchmarks
    subprocess.run(["cmake", "-DBUILD_BENCHMARKS=ON", ".."], cwd=BUILD_DIR, check=True)
    subprocess.run(['make', '-j'], cwd=BUILD_DIR, check=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmarking script for MPFX")
    parser.add_argument('output_dir', type=Path, help='Directory to save benchmark results.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for each benchmark test.')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel processes to use for benchmarking.')
    parser.add_argument('--num-inputs', type=int, default=10_000_000, help='Number of inputs per benchmark run.')
    parser.add_argument('--replot', action='store_true', help='Re-generate plots from existing benchmark data.')
    parser.add_argument('--bare', action='store_true', help='Strip the plot title and legend (e.g. for paper figures).')
    args = parser.parse_args()

    # arguments
    output_dir: Path = args.output_dir.resolve()
    iterations: int = args.iterations
    threads: int = args.threads
    num_inputs: int = args.num_inputs
    replot: bool = args.replot
    bare: bool = args.bare

    # log config
    print(f"Output Directory: {output_dir}")
    print(f"Iterations: {iterations}")
    print(f"Threads: {threads}")
    print(f"Inputs per run: {num_inputs}")
    print(f"Formats: {FORMATS}")
    print(f"Rounding modes: {RMS}")

    if not replot:
        # build benchmarks once for the whole sweep
        print('Building benchmark binaries...')
        build_benchmarks()
        print('Benchmark binaries built successfully.')

        # run every (format, rounding mode, iteration) task in one shared pool,
        # then write all aggregated results to cache
        run_all(output_dir, iterations, threads, num_inputs)

    # report each (format, rounding mode) pair from cache
    for fmt in FORMATS:
        for rm in RMS:
            report_overhead(output_dir, fmt, rm)

    # single merged speedup plot across all formats and rounding modes
    plot_speedup(output_dir, bare=bare)

    # emit LaTeX table rows (speedup vs SoftFloat)
    report_tex(output_dir)
