import csv
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

ROWS = [
    'add',
    'sub',
    'mul',
    'div',
    'sqrt',
    'fma',
]

COLUMNS = [
    'mpfr',
    'softfloat',
    'floppyfloat',
    'mpfx_rto',
    'mpfx_exact',
    'mpfx_sfloat',
    'mpfx_ffloat',
    'mpfx_eft'
]

ROUNDING_MODES = ['rne', 'rtp', 'rtz']
FORMATS = ['fp32', 'fp16']

NAMES = {
    'mpfr': 'MPFR',
    'softfloat': 'SoftFloat',
    'floppyfloat': 'FloppyFloat',
    'mpfx_rto': 'MPFX (CPU)',
    'mpfx_exact': 'MPFX (Exact)',
    'mpfx_sfloat': 'MPFX (SoftFloat)',
    'mpfx_ffloat': 'MPFX (FloppyFloat)',
    'mpfx_eft': 'MPFX (EFT)',
}


@dataclass
class TaskConfig:
    task_id: int
    cache_dir: Path
    rounding_mode: str
    format: str


def benchmark_task(config: TaskConfig):
    # run benchmark and capture output to parse as CSV
    print(f"Running benchmark task {config.task_id} (rm={config.rounding_mode}, format={config.format})...")
    benchmark_path = BUILD_DIR / "benchmark" / "ops"
    p = subprocess.run(
        [str(benchmark_path), "--rm", config.rounding_mode, "--format", config.format],
        capture_output=True, check=True
    )
    output = p.stdout.decode()

    # parse data as CSV, skipping comment lines
    data: list[list[str]] = []
    for line in output.splitlines():
        if line.startswith('#'):
            continue
        row = next(csv.reader([line]))
        data.append([datum.strip() for datum in row])

    # write raw output to cache
    cache_file = config.cache_dir / f"raw_task_{config.task_id}_rm_{config.rounding_mode}_fmt_{config.format}.csv"
    with cache_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Completed benchmark task {config.task_id} (rm={config.rounding_mode}, format={config.format}).")
    return (config.rounding_mode, config.format, data)


def run_benchmarks(output_dir: Path, iterations: int, threads: int):
    cache_dir = output_dir / "cache"
    plot_dir = output_dir / "plots"

    # Ensure directories exists
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # tasks
    configs: list[TaskConfig] = []
    task_id = 0
    for rm in ROUNDING_MODES:
        for fmt in FORMATS:
            for _ in range(iterations):
                configs.append(TaskConfig(task_id, cache_dir, rm, fmt))
                task_id += 1

    print(f"Running {len(configs)} benchmark tasks with {threads} parallel processes...")

    # run benchmarks in parallel
    # output is
    #  header0, header1, header2, ...
    #  op_name, time1, time2, ...
    results = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(benchmark_task, config) for config in configs]
        for future in futures:
            results.append(future.result())

    # group first by (rounding_mode, format, operation), then by header
    # discard the header row
    aggregated: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for rm, fmt, result in results:
        headers = result[0]
        assert headers == ['op'] + COLUMNS, f"unexpected benchmark output columns: {headers} != {['op'] + COLUMNS}"
        for row in result[1:]:
            op_name = row[0]
            key = (rm, fmt, op_name)
            if key not in aggregated:
                table: dict[str, list[float]] = {}
                for header, time in zip(headers[1:], row[1:]):
                    val = float('nan') if time == 'nan' else float(time)
                    table[header] = [val]
                aggregated[key] = table
            else:
                for header, time in zip(headers[1:], row[1:]):
                    val = float('nan') if time == 'nan' else float(time)
                    aggregated[key][header].append(val)

    print("Aggregated benchmark results.")

    # compute average time
    average_runtimes: dict[tuple[str, str, str, str], float] = {}
    for (rm, fmt, op_name), table in aggregated.items():
        for header, times in table.items():
            valid = [t for t in times if not np.isnan(t)]
            avg_time = sum(valid) / len(valid) if valid else float('nan')
            average_runtimes[(rm, fmt, op_name, header)] = avg_time

    # write average runtimes to pickle
    avg_runtime_file = cache_dir / "average_runtimes.pkl"
    with avg_runtime_file.open('wb') as f:
        pickle.dump(average_runtimes, f)

    # compute average speedup relative to SoftFloat
    average_speedups: dict[tuple[str, str, str, str], float] = {}
    for (rm, fmt, op_name), table in aggregated.items():
        baseline_times = table['softfloat']
        valid_baseline = [t for t in baseline_times if not np.isnan(t)]
        baseline = sum(valid_baseline) / len(valid_baseline) if valid_baseline else float('nan')
        for header, times in table.items():
            valid = [t for t in times if not np.isnan(t)]
            avg_time = sum(valid) / len(valid) if valid else float('nan')
            speedup = baseline / avg_time if (not np.isnan(baseline) and not np.isnan(avg_time) and avg_time != 0) else float('nan')
            average_speedups[(rm, fmt, op_name, header)] = speedup

    # write average speedups to pickle
    avg_speedup_file = cache_dir / "average_speedups.pkl"
    with avg_speedup_file.open('wb') as f:
        pickle.dump(average_speedups, f)


def report_speedup(output_dir: Path):
    # load average speedups from pickle
    avg_speedup_file = output_dir / "cache" / "average_speedups.pkl"
    with avg_speedup_file.open('rb') as f:
        average_speedups: dict[tuple[str, str, str, str], float] = pickle.load(f)

    # Create 2D table: rows are (operation, format), columns are implementations
    # Each cell contains 3 numbers for the 3 rounding modes formatted as "val1/val2/val3"
    
    # Print column suptitle
    suptitle = "Implementations (RNE/RTP/RTZ)"
    total_impl_width = 23 * len(COLUMNS)
    title_padding = (total_impl_width - len(suptitle)) // 2
    print(f"\n{'':>15}{' ' * title_padding}{suptitle}")
    
    # Print header with implementation names
    print(f"{'Operation':<15}", end="")
    for col in COLUMNS:
        display_name = NAMES.get(col, col)
        print(f"{display_name:>23}", end="")
    print()
    print("=" * (15 + 23 * len(COLUMNS)))
    
    # Print rows for each (format, operation) combination
    for idx, fmt in enumerate(FORMATS):
        for op in ROWS:
            row_label = f"{op} ({fmt.upper()})"
            print(f"{row_label:<15}", end="")
            
            # Print speedups for each implementation
            for col in COLUMNS:
                speedup_values = [average_speedups[(rm, fmt, op, col)] for rm in ROUNDING_MODES]
                speedup_str = "/".join(f"{v:.2f}" for v in speedup_values)
                print(f"{speedup_str:>23}", end="")
            print()
        
        # Add separator line between formats (except after the last one)
        if idx < len(FORMATS) - 1:
            print("-" * (15 + 23 * len(COLUMNS)))

def report_speedup_merged(output_dir: Path):
    """Like report_speedup, but merges mpfx_rto/sfloat/ffloat/eft into one column.

    Each merged cell shows the best speedup per rounding mode annotated with
    the first letter of the winning variant's qualifier: C(PU), S(oftFloat),
    F(loppyFloat), E(FT).  Format per cell: "bestRNE_X/bestRTP_Y/bestRTZ_Z".
    """
    avg_speedup_file = output_dir / "cache" / "average_speedups.pkl"
    with avg_speedup_file.open('rb') as f:
        average_speedups: dict[tuple[str, str, str, str], float] = pickle.load(f)

    MPFX_MERGE_COLS = ['mpfx_rto', 'mpfx_sfloat', 'mpfx_ffloat', 'mpfx_eft']
    MPFX_MERGE_LETTERS = {'mpfx_rto': 'C', 'mpfx_sfloat': 'S', 'mpfx_ffloat': 'F', 'mpfx_eft': 'E'}
    MERGED_COLUMNS = ['mpfr', 'softfloat', 'floppyfloat', 'mpfx_best', 'mpfx_exact']
    MERGED_NAMES = {
        'mpfr': 'MPFR',
        'softfloat': 'SoftFloat',
        'floppyfloat': 'FloppyFloat',
        'mpfx_best': 'MPFX (Best)',
        'mpfx_exact': 'MPFX (Exact)',
    }
    COL_WIDTH = 25

    suptitle = "Implementations (RNE/RTP/RTZ)"
    total_impl_width = COL_WIDTH * len(MERGED_COLUMNS)
    title_padding = (total_impl_width - len(suptitle)) // 2
    print(f"\n{'':>15}{' ' * title_padding}{suptitle}")

    print(f"{'Operation':<15}", end="")
    for col in MERGED_COLUMNS:
        print(f"{MERGED_NAMES[col]:>{COL_WIDTH}}", end="")
    print()
    print("=" * (15 + COL_WIDTH * len(MERGED_COLUMNS)))

    for idx, fmt in enumerate(FORMATS):
        for op in ROWS:
            row_label = f"{op} ({fmt.upper()})"
            print(f"{row_label:<15}", end="")
            for col in MERGED_COLUMNS:
                if col == 'mpfx_best':
                    cell_parts = []
                    for rm in ROUNDING_MODES:
                        best_val = float('nan')
                        best_letter = '?'
                        for mc in MPFX_MERGE_COLS:
                            val = average_speedups[(rm, fmt, op, mc)]
                            if np.isnan(best_val) or (not np.isnan(val) and val > best_val):
                                best_val = val
                                best_letter = MPFX_MERGE_LETTERS[mc]
                        if np.isnan(best_val):
                            cell_parts.append("nan")
                        else:
                            cell_parts.append(f"{best_val:.2f}_{best_letter}")
                    speedup_str = "/".join(cell_parts)
                else:
                    speedup_values = [average_speedups[(rm, fmt, op, col)] for rm in ROUNDING_MODES]
                    speedup_str = "/".join(f"{v:.2f}" for v in speedup_values)
                print(f"{speedup_str:>{COL_WIDTH}}", end="")
            print()

        if idx < len(FORMATS) - 1:
            print("-" * (15 + COL_WIDTH * len(MERGED_COLUMNS)))


def export_speedup_latex(output_dir: Path):
    # load average speedups from pickle
    avg_speedup_file = output_dir / "cache" / "average_speedups.pkl"
    with avg_speedup_file.open('rb') as f:
        average_speedups: dict[tuple[str, str, str, str], float] = pickle.load(f)
    
    # Find maximum speedup for each (op, format, rounding_mode) combination
    max_speedups: dict[tuple[str, str, str], dict[str, float]] = {}
    for op in ROWS:
        for fmt in FORMATS:
            for rm in ROUNDING_MODES:
                key = (op, fmt, rm)
                valid_vals = [average_speedups[(rm, fmt, op, col)] for col in COLUMNS if not np.isnan(average_speedups[(rm, fmt, op, col)])]
                max_val = max(valid_vals) if valid_vals else float('nan')
                max_speedups[key] = {'max': max_val}
    
    # Write LaTeX table to file
    latex_file = output_dir / "speedup_table.tex"
    with latex_file.open('w') as f:
        # Write table rows
        for idx, fmt in enumerate(FORMATS):
            for op in ROWS:
                row_label = f"{op} ({fmt.upper()})"
                f.write(f"{row_label}")
                
                # Write speedup values for each implementation
                for col in COLUMNS:
                    f.write(" & ")
                    
                    # Build cell content with bold formatting for maximum values
                    cell_values = []
                    for rm in ROUNDING_MODES:
                        speedup = average_speedups[(rm, fmt, op, col)]
                        max_val = max_speedups[(op, fmt, rm)]['max']

                        # Bold if this is the maximum speedup
                        if not np.isnan(speedup) and speedup == max_val:
                            cell_values.append(f"\\textbf{{{speedup:.2f}}}")
                        elif np.isnan(speedup):
                            cell_values.append("--")
                        else:
                            cell_values.append(f"{speedup:.2f}")
                    
                    f.write("/".join(cell_values))
                
                f.write(" \\\\\n")
            
            # Add separator between formats
            if idx < len(FORMATS) - 1:
                f.write("\\hline\n")
    
    print(f"Saved LaTeX table: {latex_file}")


def export_speedup_latex_merged(output_dir: Path):
    """Like export_speedup_latex, but merges mpfx_rto/sfloat/ffloat/eft into one column.

    The merged cell shows the best speedup per rounding mode with a LaTeX
    subscript letter: C(PU), S(oftFloat), F(loppyFloat), E(FT).
    Bold formatting is applied to the overall best cell in each row as usual.
    """
    avg_speedup_file = output_dir / "cache" / "average_speedups.pkl"
    with avg_speedup_file.open('rb') as f:
        average_speedups: dict[tuple[str, str, str, str], float] = pickle.load(f)

    MPFX_MERGE_COLS = ['mpfx_rto', 'mpfx_sfloat', 'mpfx_ffloat', 'mpfx_eft']
    MPFX_MERGE_LETTERS = {'mpfx_rto': 'C', 'mpfx_sfloat': 'S', 'mpfx_ffloat': 'F', 'mpfx_eft': 'E'}
    MERGED_COLUMNS = ['mpfr', 'softfloat', 'floppyfloat', 'mpfx_best', 'mpfx_exact']

    # Find maximum speedup across all merged columns for bold highlighting
    max_speedups: dict[tuple[str, str, str], float] = {}
    for op in ROWS:
        for fmt in FORMATS:
            for rm in ROUNDING_MODES:
                candidates = []
                for col in MERGED_COLUMNS:
                    if col == 'mpfx_best':
                        for mc in MPFX_MERGE_COLS:
                            val = average_speedups[(rm, fmt, op, mc)]
                            if not np.isnan(val):
                                candidates.append(val)
                    else:
                        val = average_speedups[(rm, fmt, op, col)]
                        if not np.isnan(val):
                            candidates.append(val)
                max_speedups[(op, fmt, rm)] = max(candidates) if candidates else float('nan')

    latex_file = output_dir / "speedup_table_merged.tex"
    with latex_file.open('w') as f:
        for idx, fmt in enumerate(FORMATS):
            for op in ROWS:
                row_label = f"{op} ({fmt.upper()})"
                f.write(row_label)

                for col in MERGED_COLUMNS:
                    f.write(" & ")
                    cell_values = []
                    for rm in ROUNDING_MODES:
                        max_val = max_speedups[(op, fmt, rm)]
                        if col == 'mpfx_best':
                            best_val = float('nan')
                            best_letter = '?'
                            for mc in MPFX_MERGE_COLS:
                                val = average_speedups[(rm, fmt, op, mc)]
                                if np.isnan(best_val) or (not np.isnan(val) and val > best_val):
                                    best_val = val
                                    best_letter = MPFX_MERGE_LETTERS[mc]
                            if np.isnan(best_val):
                                cell_values.append("--")
                            else:
                                subscripted = f"${best_val:.2f}_{{\\text{{{best_letter}}}}}$"
                                if not np.isnan(max_val) and best_val == max_val:
                                    cell_values.append(f"\\textbf{{{subscripted}}}")
                                else:
                                    cell_values.append(subscripted)
                        else:
                            speedup = average_speedups[(rm, fmt, op, col)]
                            if np.isnan(speedup):
                                cell_values.append("--")
                            elif not np.isnan(max_val) and speedup == max_val:
                                cell_values.append(f"\\textbf{{{speedup:.2f}}}")
                            else:
                                cell_values.append(f"{speedup:.2f}")
                    f.write("/".join(cell_values))

                f.write(" \\\\\n")

            if idx < len(FORMATS) - 1:
                f.write("\\hline\n")

    print(f"Saved merged LaTeX table: {latex_file}")


def plot_speedup(output_dir: Path):
    # load average speedups from pickle
    avg_speedup_file = output_dir / "cache" / "average_speedups.pkl"
    with avg_speedup_file.open('rb') as f:
        average_speedups: dict[tuple[str, str, str, str], float] = pickle.load(f)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create a color gradient from light to dark blue
    n_colors = len(COLUMNS)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_colors))
    
    # Create separate plots for each configuration
    for rm in ROUNDING_MODES:
        for fmt in FORMATS:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            # Create a bar chart for each operation
            for idx, op in enumerate(ROWS):
                ax: plt.Axes = axes[idx]
                
                # Get speedups for this operation and configuration
                speedups = [average_speedups[(rm, fmt, op, col)] for col in COLUMNS]
                
                # Create bar chart with gradient colors, skipping NaN
                x = np.arange(len(COLUMNS))
                bars = ax.bar(x, [s if not np.isnan(s) else 0 for s in speedups], color=colors, edgecolor='black', linewidth=0.5)
                
                # Customize plot
                ax.set_title(f'{op.upper()}', fontsize=12)
                ax.set_xticks([])  # Remove x-axis ticks and labels
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='SoftFloat baseline')
                
                # Add value labels on top of bars
                for bar, speedup in zip(bars, speedups):
                    height = bar.get_height()
                    label = f'{speedup:.2f}x' if not np.isnan(speedup) else 'nan'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label,
                           ha='center', va='bottom', fontsize=9)
            
            # Add common y-label for all subplots
            fig.supylabel('Speedup (relative to SoftFloat)', fontsize=12)
            
            # Create legend with implementation names
            legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], edgecolor='black', linewidth=0.5) 
                             for i in range(len(COLUMNS))]
            legend_labels = [NAMES.get(col, col) for col in COLUMNS]
            fig.legend(legend_patches, legend_labels, loc='center', 
                      bbox_to_anchor=(0.5, -0.02), ncol=len(COLUMNS), frameon=True,
                      fontsize=12, edgecolor='black')
            
            plt.suptitle(f'Performance Speedup by Operation (rm={rm}, format={fmt})', fontsize=16)
            plt.tight_layout(rect=[0.015, 0.03, 1, 0.96])
            
            # Save plot for this configuration
            plot_file = plot_dir / f"speedup_rm_{rm}_fmt_{fmt}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot: {plot_file}")



def report_mpfx_speedups(output_dir: Path):
    """For every (op, format, rounding mode): print MPFX(exact)/MPFR and MPFX(exact)/SoftFloat speedups."""
    avg_speedup_file = output_dir / "cache" / "average_speedups.pkl"
    with avg_speedup_file.open('rb') as f:
        average_speedups: dict[tuple[str, str, str, str], float] = pickle.load(f)

    COL_WIDTH = 22

    print(f"\nMPFX (Exact) Speedups: vs. MPFR and vs. SoftFloat")
    print(f"{'':>18}", end="")
    for rm in ROUNDING_MODES:
        header = f"[{rm.upper()}] exact/MPFR  exact/SF"
        print(f"{header:>{COL_WIDTH * 2}}", end="")
    print()
    print("=" * (18 + COL_WIDTH * 2 * len(ROUNDING_MODES)))

    for idx, fmt in enumerate(FORMATS):
        for op in ROWS:
            row_label = f"{op} ({fmt.upper()})"
            print(f"{row_label:<18}", end="")
            for rm in ROUNDING_MODES:
                sf_over_exact = average_speedups[(rm, fmt, op, 'mpfx_exact')]
                sf_over_mpfr  = average_speedups[(rm, fmt, op, 'mpfr')]

                # MPFX (exact) speedup over MPFR
                if not np.isnan(sf_over_exact) and not np.isnan(sf_over_mpfr) and sf_over_mpfr != 0:
                    exact_over_mpfr = sf_over_exact / sf_over_mpfr
                    mpfr_str = f"{exact_over_mpfr:.2f}x"
                else:
                    mpfr_str = "nan"

                # MPFX (exact) speedup over SoftFloat (already relative to SF in the cache)
                sf_str = f"{sf_over_exact:.2f}x" if not np.isnan(sf_over_exact) else "nan"

                cell = f"{mpfr_str:>10}  {sf_str:<10}"
                print(f"{cell:>{COL_WIDTH * 2}}", end="")
            print()

        if idx < len(FORMATS) - 1:
            print("-" * (18 + COL_WIDTH * 2 * len(ROUNDING_MODES)))


def build_benchmarks():
    # Navigate to build directory and build benchmarks
    subprocess.run(["cmake", "-DBUILD_BENCHMARKS=ON", ".."], cwd=BUILD_DIR, check=True)
    subprocess.run(['make', '-j'], cwd=BUILD_DIR, check=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmarking script for MPFX")
    parser.add_argument('output_dir', type=Path, help='Directory to save benchmark results.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for each benchmark test.')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel processes to use for benchmarking.')
    parser.add_argument('--replot', action='store_true', help='Re-generate plots from existing benchmark data.')
    args = parser.parse_args()

    # arguments
    output_dir: Path = args.output_dir.resolve()
    iterations: int = args.iterations
    threads: int = args.threads
    replot: bool = args.replot

    # log config
    print(f"Output Directory: {output_dir}")
    print(f"Iterations: {iterations}")
    print(f"Threads: {threads}")

    if not replot:
        # build benchmarks
        print('Building benchmark binaries...')
        build_benchmarks()
        print('Benchmark binaries built successfully.')

        # Run benchmarks
        run_benchmarks(output_dir, iterations, threads)
    
    # Report speedups
    report_speedup(output_dir)
    report_speedup_merged(output_dir)
    report_mpfx_speedups(output_dir)

    # Export speedup table to LaTeX
    export_speedup_latex(output_dir)
    export_speedup_latex_merged(output_dir)

    # Plot speedup
    plot_speedup(output_dir)
