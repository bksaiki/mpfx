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
    'native',
    'mpfr',
    'softfloat',
    'floppyfloat',
    'mpfx_rto',
    'mpfx_sfloat',
    'mpfx_ffloat',
    'mpfx_eft'
]

NAMES = {
    'mpfr': 'MPFR',
    'softfloat': 'SoftFloat',
    'floppyfloat': 'FloppyFloat',
    'mpfx_rto': 'MPFX (RTO)',
    'mpfx_sfloat': 'MPFX (SoftFloat)',
    'mpfx_ffloat': 'MPFX (FloppyFloat)',
    'mpfx_eft': 'MPFX (EFT)',
}


@dataclass
class TaskConfig:
    task_id: int
    cache_dir: Path


def benchmark_task(config: TaskConfig):
    # run benchmark and capture output to parse as CSV
    print(f"Running benchmark task {config.task_id}...")
    benchmark_path = BUILD_DIR / "benchmark" / "ops"
    p = subprocess.run([str(benchmark_path)], capture_output=True, check=True)
    output = p.stdout.decode()

    # parse data as CSV
    data: list[list[str]] = []
    for row in csv.reader(output.splitlines()):
        data.append([datum.strip() for datum in row])

    # write raw output to cache
    cache_file = config.cache_dir / f"raw_task_{config.task_id}.csv"
    with cache_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Completed benchmark task {config.task_id}.")
    return data


def run_benchmarks(output_dir: Path, iterations: int, threads: int):
    cache_dir = output_dir / "cache"
    plot_dir = output_dir / "plots"

    # Ensure directories exists
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # run benchmarks in parallel
    # output is
    #  header0, header1, header2, ...
    #  op_name, time1, time2, ...
    results = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        configs: list[TaskConfig] = []
        for i in range(iterations):
            configs.append(TaskConfig(i, cache_dir))

        futures = [executor.submit(benchmark_task, config) for config in configs]
        for future in futures:
            results.append(future.result())

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
                    table[header] = [float(time)]
                aggregated[op_name] = table
            else:
                for header, time in zip(headers[1:], row[1:]):
                    aggregated[op_name][header].append(float(time))

    print("Aggregated benchmark results.")

    # compute average time
    average_runtimes: dict[tuple[str, str], float] = {}
    for op_name, table in aggregated.items():
        for header, times in table.items():
            avg_time = sum(times) / len(times)
            average_runtimes[(op_name, header)] = avg_time

    # write average runtimes to pickle
    avg_runtime_file = cache_dir / "average_runtimes.pkl"
    with avg_runtime_file.open('wb') as f:
        pickle.dump(average_runtimes, f)

    # compute average overhead over native
    average_overheads: dict[tuple[str, str], float] = {}
    for op_name, table in aggregated.items():
        baseline_times = table['native']
        baseline = sum(baseline_times) / len(baseline_times)
        for header, times in table.items():
            if header != 'native':
                avg_time = sum(times) / len(times)
                overhead = avg_time / baseline
                average_overheads[(op_name, header)] = overhead

    # write average overheads to pickle
    avg_overhead_file = cache_dir / "average_overheads.pkl"
    with avg_overhead_file.open('wb') as f:
        pickle.dump(average_overheads, f)


def report_overhead(output_dir: Path):
    # load average overheads from pickle
    avg_overhead_file = output_dir / "cache" / "average_overheads.pkl"
    with avg_overhead_file.open('rb') as f:
        average_overheads: dict[tuple[str, str], float] = pickle.load(f)

    print(f'{"op":<12}', end="")
    for col in COLUMNS[1:]:
        print(f"{col:>12}", end="")
    print()

    for row in ROWS:
        print(f"{row:<12}", end="")
        for col in COLUMNS[1:]:
            overhead = average_overheads[(row, col)]
            print(f'{overhead:>12.2f}', end="")
        print()

def plot_overhead(output_dir: Path):
    # load average overheads from pickle
    avg_overhead_file = output_dir / "cache" / "average_overheads.pkl"
    with avg_overhead_file.open('rb') as f:
        average_overheads: dict[tuple[str, str], float] = pickle.load(f)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create a color gradient from light to dark blue
    n_colors = len(COLUMNS[1:])
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_colors))
    
    # Create a single figure with subplots for all operations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Create a bar chart for each operation
    for idx, op in enumerate(ROWS):
        ax: plt.Axes = axes[idx]
        
        # Get overheads for this operation
        overheads = [average_overheads[(op, col)] for col in COLUMNS[1:]]
        
        # Create bar chart with gradient colors
        x = np.arange(len(COLUMNS[1:]))
        bars = ax.bar(x, overheads, color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_title(f'{op.upper()}', fontsize=12)
        ax.set_xticks([])  # Remove x-axis ticks and labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}x',
                   ha='center', va='bottom', fontsize=10)
    
    # Add common y-label for all subplots
    fig.supylabel('Overhead (relative to native)', fontsize=12)
    
    # Create legend with implementation names
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], edgecolor='black', linewidth=0.5) 
                     for i in range(len(COLUMNS[1:]))]
    legend_labels = [NAMES[col] for col in COLUMNS[1:]]
    fig.legend(legend_patches, legend_labels, loc='center', 
              bbox_to_anchor=(0.5, -0.02), ncol=len(COLUMNS[1:]), frameon=True,
              fontsize=12, edgecolor='black')
    
    plt.suptitle('Performance Overhead by Operation', fontsize=16)
    plt.tight_layout(rect=[0.015, 0.03, 1, 0.96])
    
    # Save combined plot
    plot_file = plot_dir / "overhead.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {plot_file}")



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

    # Report overheads
    report_overhead(output_dir)

    # Plot overhead
    plot_overhead(output_dir)
