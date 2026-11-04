import csv
import subprocess

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
BUILD_DIR = REPO_ROOT / "build"
BINARY = BUILD_DIR / "examples" / "mx_dot_prod"


def parse_seconds(s: str) -> float:
    # timing cells look like `1.1551s`
    return float(s.strip().removesuffix('s'))


def run_iteration(iteration: int, num_inputs: int) -> list[tuple[str, float, float]]:
    # `mx_dot_prod <N> [seed]` seeds its inputs (1 by default), so seed each
    # iteration differently to sample a different input set per run.
    print(f"Running iteration {iteration}...")
    p = subprocess.run(
        [str(BINARY), str(num_inputs), str(iteration + 1)],
        capture_output=True,
        check=True,
    )

    # The timing table looks like:
    #   Config        | SoftFloat     | MPFX          | Speedup       |
    #   --------------|---------------|---------------|---------------|
    #   E5M2xE5M2     | 1.1551s       | 0.1138s       | 10.15x        |
    # Skip progress lines, the header, and the separator; keep data rows.
    rows: list[tuple[str, float, float]] = []
    for line in p.stdout.decode().splitlines():
        cells = [c.strip() for c in line.split('|')]
        if len(cells) < 3 or cells[0] == 'Config' or cells[0].startswith('-'):
            continue
        rows.append((cells[0], parse_seconds(cells[1]), parse_seconds(cells[2])))

    print(f"Completed iteration {iteration}.")
    return rows


def build_example():
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(["cmake", "-DBUILD_EXAMPLES=ON", ".."], cwd=BUILD_DIR, check=True)
    subprocess.run(['make', '-j', 'mx_dot_prod'], cwd=BUILD_DIR, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Averages `mx_dot_prod` timings over multiple runs")
    parser.add_argument('output_dir', type=Path, help='Directory to save results.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of runs to average over.')
    parser.add_argument('--threads', type=int, default=1, help='Number of runs to launch in parallel.')
    parser.add_argument('--num-inputs', type=int, default=100_000, help='Number of dot products per run.')
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    iterations: int = args.iterations
    threads: int = args.threads
    num_inputs: int = args.num_inputs

    print(f"Output Directory: {output_dir}")
    print(f"Iterations: {iterations}")
    print(f"Threads: {threads}")
    print(f"Inputs per run: {num_inputs}")

    print('Building mx_dot_prod...')
    build_example()

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(run_iteration, i, num_inputs) for i in range(iterations)]
        results = [future.result() for future in futures]

    # collect times per config, preserving the binary's config order
    sf_times: dict[str, list[float]] = {}
    mpfx_times: dict[str, list[float]] = {}
    for rows in results:
        for config, sf_time, mpfx_time in rows:
            sf_times.setdefault(config, []).append(sf_time)
            mpfx_times.setdefault(config, []).append(mpfx_time)

    # write every raw measurement
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_file = output_dir / "raw.csv"
    with raw_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'config', 'softfloat', 'mpfx'])
        for i, rows in enumerate(results):
            for config, sf_time, mpfx_time in rows:
                writer.writerow([i, config, sf_time, mpfx_time])

    # speedup is the ratio of average times, matching `benchmark.py`
    summary: list[tuple[str, float, float, float]] = []
    for config in sf_times:
        sf_avg = sum(sf_times[config]) / len(sf_times[config])
        mpfx_avg = sum(mpfx_times[config]) / len(mpfx_times[config])
        summary.append((config, sf_avg, mpfx_avg, sf_avg / mpfx_avg))

    summary_file = output_dir / "summary.csv"
    with summary_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'softfloat', 'mpfx', 'speedup'])
        writer.writerows(summary)

    print(f"\n# Average over {iterations} runs of {num_inputs} inputs")
    print(f'{"config":<14}{"softfloat":>12}{"mpfx":>12}{"speedup":>10}')
    for config, sf_avg, mpfx_avg, speedup in summary:
        print(f"{config:<14}{sf_avg:>11.4f}s{mpfx_avg:>11.4f}s{speedup:>9.2f}x")

    print(f"\nSaved raw results: {raw_file}")
    print(f"Saved summary: {summary_file}")
