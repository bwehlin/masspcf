#!/usr/bin/env bash
# Quick multi-GPU / scaling sanity sweep for the hybrid Ripser++ dispatcher.
#
# Exercises three regimes:
#   * n=1500 max_dim=1 batch=128  -- GPU-friendly but host-dominated,
#                                    tests multi-GPU admits + CPU pool ceiling
#   * n=500  max_dim=2 batch=64   -- paper's apparent-pairs sweet spot,
#                                    tests per-card 5-7x GPU advantage
#   * n=5000 max_dim=1 batch=128  -- large n so per-item GPU kernels dominate,
#                                    tests actual GPU saturation
#
# Run from repo root, inside the correct Python env, after a release
# build + install:
#   cmake --build cmake-build-release -j$(nproc) && cmake --install cmake-build-release
#   ./benchmarks/run_multigpu_check.sh
#
# Logs go to /tmp/a4000_maxdim{1,2}.log and to benchmarks/bench_results/*.{csv,png}.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "# Run 1: n=1500 max_dim=1 batch=128 (multi-GPU admit + thread-pool ceiling)"
python -u benchmarks/bench_ph_hybrid.py cooperate \
  --n 1500 --max-dim 1 --batch-size 128 --gpu-caps 8 16 32 0 \
  | tee /tmp/a4000_maxdim1.log

echo
echo "# Run 2: n=500 max_dim=2 batch=64 (apparent-pairs sweet spot)"
python -u benchmarks/bench_ph_hybrid.py cooperate \
  --n 500 --max-dim 2 --batch-size 64 --gpu-caps 4 8 16 0 \
  | tee /tmp/a4000_maxdim2.log

echo
echo "# Run 3: n=5000 max_dim=1 batch=128 (large n, GPU-kernel-dominated)"
python -u benchmarks/bench_ph_hybrid.py cooperate \
  --n 5000 --max-dim 1 --batch-size 128 --gpu-caps 16 32 0 \
  | tee /tmp/a4000_bigger.log

echo
echo "# Done. Summary:"
echo "#   /tmp/a4000_maxdim1.log"
echo "#   /tmp/a4000_maxdim2.log"
echo "#   /tmp/a4000_bigger.log"
echo "#   benchmarks/bench_results/index.html   (open in browser for plots)"
