# Get directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running benchmark_add..."
"$SCRIPT_DIR"/build/examples/benchmark_add

echo "Running benchmark_mul..."
"$SCRIPT_DIR"/build/examples/benchmark_mul

echo "Running benchmark_div..."
"$SCRIPT_DIR"/build/examples/benchmark_div

echo "Running benchmark_fma..."
"$SCRIPT_DIR"/build/examples/benchmark_fma

echo "Running benchmark_sqrt..."
"$SCRIPT_DIR"/build/examples/benchmark_sqrt

echo "Running benchmark_add_engines..."
"$SCRIPT_DIR"/build/examples/benchmark_add_engines

echo "Running benchmark_mul_engines..."
"$SCRIPT_DIR"/build/examples/benchmark_mul_engines

echo "Running benchmark_div_engines..."
"$SCRIPT_DIR"/build/examples/benchmark_div_engines

echo "Running mixed_dot_prod..."
"$SCRIPT_DIR"/build/examples/mixed_dot_prod
