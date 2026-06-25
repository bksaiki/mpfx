# exit on error
set -e

# Get directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running benchmark_ops..."
"$SCRIPT_DIR"/build/benchmark/benchmark_ops

echo "Running benchmark_round..."
"$SCRIPT_DIR"/build/benchmark/benchmark_round

echo "Running mixed_dot_prod..."
"$SCRIPT_DIR"/build/examples/mixed_dot_prod
