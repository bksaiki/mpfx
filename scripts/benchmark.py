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

RNE, RTP, RAZ = 'rne', 'rtp', 'raz'
RMS = [RNE, RTP, RAZ]

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
    'mpfx_sfloat',
    'mpfx_ffloat',
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
        assert headers[1:] == COLUMNS, f"unexpected benchmark output columns: {headers[1:]} != {COLUMNS}"
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


def plot_overhead(output_dir: Path, fmt: str, rm: str):
    # load average overheads from pickle
    avg_overhead_file = output_dir / "cache" / f"average_overheads_{fmt}_{rm}.pkl"
    with avg_overhead_file.open('rb') as f:
        average_overheads: dict[tuple[str, str], float] = pickle.load(f)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create a color gradient from light to dark blue
    n_colors = len(DISPLAY_COLUMNS)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_colors))

    # Create a single figure with subplots for all operations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Create a bar chart for each operation
    for idx, op in enumerate(ROWS):
        ax: plt.Axes = axes[idx]

        # Get overheads for this operation
        overheads = [average_overheads[(op, col)] for col in DISPLAY_COLUMNS]

        # Create bar chart with gradient colors
        x = np.arange(len(DISPLAY_COLUMNS))
        bars = ax.bar(x, overheads, color=colors, edgecolor='black', linewidth=0.5)

        # Reference line at the baseline (1.0)
        ax.axhline(1.0, color='red', linewidth=0.8, linestyle='--', alpha=0.6)

        # Customize plot
        ax.set_title(f'{op.upper()}', fontsize=12)
        ax.set_xticks([])  # Remove x-axis ticks and labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on top of bars ("n/a" for columns that were not run)
        for bar, overhead in zip(bars, overheads):
            if math.isnan(overhead):
                ax.text(bar.get_x() + bar.get_width()/2., 0.0,
                        'n/a', ha='center', va='bottom', fontsize=9, color='gray')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{overhead:.1f}x',
                        ha='center', va='bottom', fontsize=10)

    # Add common y-label for all subplots
    fig.supylabel(f'Runtime relative to {NAMES[BASELINE]}', fontsize=12)

    # Create legend with implementation names
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], edgecolor='black', linewidth=0.5)
                     for i in range(len(DISPLAY_COLUMNS))]
    legend_labels = [NAMES[col] for col in DISPLAY_COLUMNS]
    fig.legend(legend_patches, legend_labels, loc='center',
              bbox_to_anchor=(0.5, -0.02), ncol=len(DISPLAY_COLUMNS), frameon=True,
              fontsize=12, edgecolor='black')

    plt.suptitle(f'Performance Overhead by Operation ({fmt}/{rm}, relative to {NAMES[BASELINE]})', fontsize=16)
    plt.tight_layout(rect=[0.015, 0.03, 1, 0.96])

    # Save combined plot
    plot_file = plot_dir / f"overhead_{fmt}_{rm}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
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
    args = parser.parse_args()

    # arguments
    output_dir: Path = args.output_dir.resolve()
    iterations: int = args.iterations
    threads: int = args.threads
    num_inputs: int = args.num_inputs
    replot: bool = args.replot

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

    # report and plot each (format, rounding mode) pair from cache
    for fmt in FORMATS:
        for rm in RMS:
            report_overhead(output_dir, fmt, rm)
            plot_overhead(output_dir, fmt, rm)

    # emit LaTeX table rows (speedup vs SoftFloat)
    report_tex(output_dir)
