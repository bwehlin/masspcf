# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**masspcf** is a Python package with a C++20/CUDA backend for massively parallel computation of similarity matrices from piecewise functions (currently piecewise constant functions, with piecewise linear functions planned). The primary audience is TDA (Topological Data Analysis) practitioners doing statistical analysis on invariants such as stable rank, Euler characteristic curves, and Betti curves. The core objects are numpy-like multidimensional arrays (tensors) supporting reductions, Lp distance matrices, and L2 kernels.

## Build & Development

### Full install (required before minimal builds)
```bash
pip install .
```

### Minimal module build (for iterative C++ development)
```bash
cmake -B cmake-build-debug
cmake --build cmake-build-debug -j$(nproc)
cmake --install cmake-build-debug
```
This works when `SKBUILD` is off (plain CMake). Builds `_mpcf_cpu` (always) and `_mpcf_cudaXX` (when CUDA available). Extensions are symlinked into `masspcf/` for immediate use.

### CUDA control
- `BUILD_WITH_CUDA=0` env var disables CUDA (auto-detected otherwise, always off on macOS)
- Supports pip-installed CUDA toolkits (`nvidia.cu12`, `nvidia.cu13`)
- Produces version-specific modules: `_mpcf_cuda12`, `_mpcf_cuda13`

### Other env vars
- `SKIP_STUBGEN=1` — skip pybind11_stubgen (useful if it causes issues)
- `SKIP_BACK_COPY=1` — skip symlinking built extensions back to source tree
- `ENABLE_COVERAGE=1` — build with coverage instrumentation (GCC or Clang)

## Testing

**Important**: Always `cd test` before running pytest. Running from the repo root causes the local `masspcf/` directory to shadow the installed package. You must also build and install first.

### Python tests
```bash
cmake --build cmake-build-debug -j$(nproc) && cmake --install cmake-build-debug
cd test && python -m pytest python               # all Python tests
cd test && python -m pytest python/test_pdist.py  # single test file
```

### C++ tests (GoogleTest)
```bash
cmake --build cmake-build-debug --target mpcf_test
cd test && ../cmake-build-debug/mpcf_test  # run from test/ directory
```

### Coverage
```bash
./run_coverage.sh          # full local coverage report (C++ + Python)
./run_coverage.sh --open   # open HTML reports in browser
```

## Architecture

### Python ↔ C++ boundary
- **`masspcf/_mpcf_cpp.py`** is the runtime backend selector. It detects GPU availability, preloads libcudart if needed, and imports `_mpcf_cuda{12,13}` or falls back to `_mpcf_cpu`. All other Python code imports through this module.
- **`src/python/`** contains pybind11 bindings. Each `py_*.cpp` wraps a corresponding C++ subsystem.
- **`src/gpu_detect/`** is a separate pybind11 module (`_gpu_detect`) with no CUDA dependency — used to detect GPUs without requiring the CUDA toolkit.
- NumPy array ↔ C++ tensor conversion happens in `py_np_tensor_convert.{h,cpp}` (zero-copy where possible).

### C++ core (`include/mpcf/`)
- **`pcf.h`** — `Pcf<TimeT, ValueT>` template, the fundamental piecewise constant function type
- **`tensor.h` / `tensor.tpp`** — N-dimensional tensor template (stores PCFs, floats, point clouds, or barcodes)
- **`algorithms/`** — core algorithms: `matrix_integrate.h` (distance matrices), `reduce.h`, `iterate_rectangles.h`, `subdivide.h`, `apply_functional.h`
- **`persistence/`** — TDA: ripser algorithm, barcodes, stable rank
- **`cuda/`** — CUDA kernels for matrix operations
- **`executor.h`** — taskflow-based parallel task dispatch (CPU)

### Type system
Each tensor class supports multiple precisions via a `dtype` parameter: `PcfTensor` (dtype=pcf32/pcf64), `IntPcfTensor` (dtype=pcf32i/pcf64i), `FloatTensor` (dtype=float32/float64), `PointCloudTensor` (dtype=pcloud32/pcloud64), `BarcodeTensor` (dtype=barcode32/barcode64), `DistanceMatrixTensor` (dtype=distmat32/distmat64), `SymmetricMatrixTensor` (dtype=symmat32/symmat64). Dtype sentinels live in `masspcf/typing.py`. The C++ layer still has separate types per precision (e.g. `cpp.Float32Tensor`, `cpp.Float64Tensor`); the Python classes dispatch internally.

### Python module layers
1. **Low**: `_mpcf_cpp` (backend dispatch) → `_mpcf_cpu` / `_mpcf_cudaXX` (pybind11)
2. **Mid**: `pcf.py`, `tensor.py`, `_tensor_base.py` (Python wrappers)
3. **High**: `distance.py` (`pdist`), `reductions.py` (`mean`, `max_time`), `norms.py`, `persistence/`, `timeseries/`, `sklearn/`

### TimeSeries (`masspcf/timeseries/`)
`TimeSeries` wraps a PCF with real-world time metadata (start_time, time_step). Supports float and `datetime64` time axes, nearest and linear interpolation, multi-channel data (channels-first: shape `(n_channels, n_times)`), and save/load/pickle. `TimeSeriesTensor` is the corresponding N-dimensional container. `embed_time_delay()` computes Takens-style time delay embeddings, producing `PointCloudTensor` output with optional windowing. The C++ core lives in `include/mpcf/timeseries.hpp` and `include/mpcf/algorithms/embed_time_delay.hpp`; pybind11 bindings in `src/python/py_timeseries.cpp`.

### sklearn integration (`masspcf/sklearn/`)
Optional subpackage (requires scikit-learn). Transformers: `TimeDelayEmbedding`, `PersistentHomology`, `StableRank`, `Mean`, `PcfKernelTransformer`. All use `no_validation=True` and non-standard input shapes. `PcfKernelTransformer.fit_transform` has an optimized symmetric-kernel path distinct from `fit` + `transform`.

### GPU/CPU runtime control (`masspcf/system.py`)
`force_cpu()`, `limit_cpus()`, `limit_gpus()`, `set_cuda_threshold()`, `set_device_verbose()`, `set_parallel_eval_threshold()` — all configure the backend at runtime.

## Engineering philosophy

Be direct and push back when something doesn't make sense or takes unnecessary shortcuts. Don't apply band-aid fixes — investigate and address root causes, even if that means more work. Ask before proceeding if the scope grows significantly, but be ready to execute on it. Correctness over velocity; let things take longer if they need to.

When writing tests, think about the behaviors the code implements and the edge cases around them. Don't write tests that just verify "it ran and didn't crash," and don't chase 100% coverage through meaningless tests. Every test should assert something specific about a behavior.

## Third-party submodules (`3rd/`)
- **pybind11** — Python/C++ bindings
- **taskflow** — CPU task parallelism (header-only, included directly)
- **googletest** — C++ unit tests

## Documentation (`docs/`)
- Sphinx docs live in `docs/`, built with `make html` from that directory.
- **Keep docs in sync with code changes.** When renaming parameters, changing defaults, or modifying public API behavior, update the corresponding `.rst` files and docstrings in the same commit.
- **Document new public functionality.** New public APIs, features, or user-visible behavior changes should be documented in the appropriate `.rst` file unless purely internal.
- **Plots with code dropdowns.** When adding plots/figures to docs, always include a `.. dropdown:: Show code` with a `.. literalinclude::` referencing snippet markers in the plot generation script. Plot generation scripts live in `docs/_static/`.

## Key files
- `pyproject.toml` — single source of truth for version (`[project].version`), dependencies, and wheel build config
- `version.cpp.in` — CMake template that embeds version + build date into the binary
- `.clang-format` — C++ formatting (Microsoft style)
- `.codacy.yaml` — static analysis config (excludes `3rd/`)
