# Ripser++ integration plan

Working branch: `dev`. Upstream: https://github.com/simonzhang00/ripser-plusplus (MIT).

## Goal

Add GPU-accelerated persistent homology via a ported Ripser++ alongside the
existing CPU Ripser, with a hybrid dispatcher that feeds items to whichever
side is free (CPU workers + GPU lane(s)). On GPU OOM, a worker falls back to
the CPU path for that item rather than re-queueing.

## Context / why

- Baseline scaling of the current CPU Ripser on `samples_3_50_2500.mpcf`
  (150 point clouds × 2500 pts, 36-core box):
  - W=8  → 49.8 s (3.01 items/s, per-item 2.66 s serial-equiv)
  - W=18 → 28.7 s (5.23 items/s)
  - W=30 → 24.6 s (6.10 items/s)
  - W=36 → 24.4 s (6.16 items/s)
- Per-item serial-equivalent time more than doubles 8 → 36 workers:
  each Ripser allocates big distance / reduction matrices, and at 36
  concurrent instances we're memory-bandwidth-bound, not compute-bound.
- The visible "CPU utilization drop" during a run is the tail: 150 / 36
  is 4 r 6, so 30 cores idle for the last ~4 s.
- Partitioner choice (`GuidedPartitioner` default vs `DynamicPartitioner<>(1)`)
  made no measurable difference on uniform-size items. Both revert-benches
  landed within noise.
- Conclusion: further CPU parallelism is capped by memory bandwidth; GPU is
  the next meaningful speedup lever.

## Design decisions (agreed)

1. **Hybrid dispatcher, not a separate task.** One unified
   `compute_persistent_homology` API; items flow to whichever worker
   (CPU or GPU) grabs them next via a shared atomic counter.
2. **Allow multiple concurrent GPU jobs.** Per-instance CUDA streams;
   `thrust::cuda::par.on(stream)` threaded through every thrust call site
   in the port (phase 3 below).
3. **No memory pre-estimation.** On GPU-side OOM, catch the exception,
   fall back to CPU for that one item. Simpler than re-queue/backoff; avoids
   thrashing.
4. **No `exit()` calls.** Replace `CUDACHECK` with a throwing wrapper;
   remove `main()` / `run_main()` / `run_main_filename()` /
   `print_usage_and_exit()` entirely. Every error path must either throw
   or propagate an error code.
5. **Layout**: Ripser++ code stays in a `.cu` (has `__global__` kernels).
   Bundled header-only deps (phmap, sparsehash) go under
   `include/mpcf/persistence/ripserpp/` so C++ consumers that only need
   headers aren't missing transitive deps. Main `.cu` lives in `src/cuda/`.

## Layout

```
3rd/ripserpp/LICENSE                                         # MIT attribution (upstream)  [DONE]
include/mpcf/persistence/ripserpp/
    parallel_hashmap/*.h                                     # Apache 2.0, header-only    [DONE]
    sparsehash/*                                             # BSD, header-only           [DONE]
    phmap_interface.hpp                                      # inlined shim               [DONE]
    ripserpp.hpp                                             # public facade API          [TODO]
src/cuda/ripserpp.cu                                         # ported main source         [copied verbatim]
```

The `profiling/stopwatch.h` dep from upstream is LGPL — do **not** vendor it.
Replace uses with a tiny `std::chrono` wrapper (or inline `#ifdef PROFILING`
blocks that guard all its uses already).

## Known thread-safety issues to fix

These are blockers for running more than one Ripser++ invocation simultaneously
on the GPU:

1. **`list_of_barcodes` global** in `src/cuda/ripserpp.cu`
   (file-scope `std::vector<std::vector<birth_death_coordinate>>`).
   Every `ripser<...>::compute_barcodes` push_back's into this global.
   **Fix**: thread an output collector (reference / pointer) through the
   `ripser` template class so each instance writes to its own container.
   Touch sites: definitions near line 175; use sites around lines 1908, 2089,
   2187, 2414, 2496 (dim-0 + higher-dim append paths in both
   `compressed_lower_distance_matrix` and `sparse_distance_matrix`
   specializations).

2. **`phmap_interface` global hash map** (now in
   `include/mpcf/persistence/ripserpp/phmap_interface.hpp`, was a file-scope
   `phmap::parallel_flat_hash_map<int64_t,int64_t>` in upstream's .cpp).
   The inlined header keeps the singleton for now — explicitly marked in the
   file header as not thread-safe.
   **Fix**: make it an instance member of the `ripser` class (or a context
   object passed in), not a singleton. Upstream uses it as a scratch pivot
   column map inside one invocation, so it's a natural per-instance resource.
   Audit `phmap_put` / `phmap_get_value` / `phmap_clear` call sites in
   `ripserpp.cu` and route them through the instance.

3. **`cudaSetDevice` / default stream usage.** All thrust calls currently go
   on the default stream. For concurrent GPU instances to overlap, each
   `ripser` instance needs its own `cudaStream_t` and every thrust call needs
   `thrust::cuda::par.on(stream)`. Raw kernel launches also need `<<<..., stream>>>`.

4. **Raw `cudaMalloc` without RAII.** Audit needed — exceptions thrown from
   mid-computation would leak device memory. Wrap all raw cudaMalloc blocks
   in an RAII guard (or replace with `thrust::device_vector`).

5. **`exit(EXIT_FAILURE)` / `exit(1)` / `exit(-1)`** — 39 call sites in the
   original `.cu`. All must be replaced with `throw std::runtime_error(...)`
   (or a dedicated `mpcf::ripserpp::cuda_error` / `oom_error` type so the
   hybrid dispatcher can catch OOM specifically for CPU fallback).
   - Line 49 (`CUDACHECK` macro itself — the biggest one, replace with
     throwing macro, and make the CUDA error type distinguishable from
     generic runtime errors so dispatcher can catch OOM).
   - Scattered `exit(1)` / `exit(-1)` across arg-parsing and validation code
     in `run_main`, `run_main_filename`, `print_usage_and_exit` — these
     whole functions are deleted, not patched.
   - Validation `exit(1)`s inside `compute_barcodes` paths (around lines
     3088, 3114, 3137, 3291, 3337, 3365, 3374) → throw.

6. **5× `#pragma omp parallel for schedule(guided,1)`** (lines 2425, 2507,
   2705, 2721, 2821). Replace with `tf::Taskflow::for_each_index` using the
   project's `Executor`. The loops are inside `compute_barcodes` so each
   call needs access to an executor — thread it through the `ripser`
   constructor or pass an `mpcf::Executor&` to `compute_barcodes`.

## Step-by-step plan

### Phase 1 — verbatim lift, make it build [partially done]

- [x] `3rd/ripserpp/LICENSE`
- [x] `include/mpcf/persistence/ripserpp/{parallel_hashmap,sparsehash}/` — copied
- [x] `include/mpcf/persistence/ripserpp/phmap_interface.hpp` — header-only shim
- [x] `src/cuda/ripserpp.cu` — copied from upstream (unpatched)
- [ ] Patch `src/cuda/ripserpp.cu`:
  - Update includes:
    - `#include <parallel_hashmap/phmap.h>` →
      `#include <mpcf/persistence/ripserpp/parallel_hashmap/phmap.h>`
    - `#include <sparsehash/dense_hash_map>` →
      `#include <mpcf/persistence/ripserpp/sparsehash/dense_hash_map>`
    - `#include <phmap_interface/phmap_interface.h>` →
      `#include <mpcf/persistence/ripserpp/phmap_interface.hpp>`
    - `#include <profiling/stopwatch.h>` — [x] removed; all `Stopwatch` uses
      and their associated `#ifdef PROFILING` timing printouts stripped from
      `src/cuda/ripserpp.cu` (fast-tracked ahead of the rest of phase 1).
  - Replace `CUDACHECK`:
    ```cpp
    #define CUDACHECK(cmd) do { \
      cudaError_t _e = (cmd); \
      if (_e != cudaSuccess) { \
        throw ::mpcf::ripserpp::cuda_error(__FILE__, __LINE__, _e); \
      } \
    } while (0)
    ```
    Define `cuda_error` (with a `bool is_oom() const` helper checking
    `cudaErrorMemoryAllocation`) in `include/mpcf/persistence/ripserpp/ripserpp.hpp`.
  - Delete `main()` (line ~4050), `run_main()` (~3887), `run_main_filename()`
    (~3720), `print_usage_and_exit()` (~3691), and related
    `ripser_plusplus_result` C-API structs if they're only used by those.
  - Replace every remaining `exit(...)` with `throw std::runtime_error(...)`.
- [ ] Add CMake target: new library (or objects) compiled with nvcc, linked
  into `_mpcf_cuda12` / `_mpcf_cuda13`. Guard under `BUILD_WITH_CUDA`.
  The existing `CMakeLists.txt` already has the pattern — model after
  `cuda_matrix_integrate.cu`.
- [ ] Stub facade `include/mpcf/persistence/ripserpp/ripserpp.hpp` exposing
  a single C++ function, something like:
  ```cpp
  namespace mpcf::ph::ripserpp {
    template <typename T>
    void compute_barcodes_pcloud(
        const PointCloud<T>& points,
        size_t maxDim,
        std::vector<std::vector<PersistencePair<T>>>& out);   // out sized maxDim+1
  }
  ```
- [ ] Verify it **builds** (no functional use yet).

### Phase 2 — first functional use, single GPU job

- [ ] Thread `list_of_barcodes` through `compute_barcodes` as an output
  parameter (kills global #1).
- [ ] Replace OMP pragmas with taskflow `for_each_index` — needs an
  `mpcf::Executor&` threaded in (minor API addition).
- [ ] Make the phmap_interface singleton an instance member of `ripser`
  (kills global #2).
- [ ] Add `RipserPlusPlusTask<T>` in
  `include/mpcf/persistence/compute_persistence.hpp`, mirroring
  `RipserTaskImpl` but iterating items **serially** on the GPU for now.
- [ ] pybind: new `spawn_ripser_plusplus_*_task` entries, gated so the
  symbols only exist in `_mpcf_cudaXX`.
- [ ] Python: route through `compute_persistent_homology(..., device="gpu")`
  or auto-detect via `system.cuda_available()`.
- [ ] Correctness test: compare barcodes against CPU Ripser on the stable
  existing test cases (`test/python/persistence/test_ripser.py`), within
  tolerance. Ripser++ reduces homology the same way — expect identical
  pairs modulo ordering. The H0-essential-class handling (we add `[0, inf)`
  unreduced) needs to match.

### Phase 3 — hybrid dispatcher + concurrent GPU jobs

- [ ] Introduce shared atomic-counter dispatch primitive in
  `include/mpcf/dispatch.hpp` (or inline it in compute_persistence).
  Shape:
  ```cpp
  std::atomic<size_t> next{0};
  // N CPU workers + M GPU workers (M discovered at runtime; start at 1)
  // each loops: i = next.fetch_add(1); if (i >= total) break;
  //             try { process_gpu(i) } catch (oom) { process_cpu(i) }
  ```
- [ ] Audit raw `cudaMalloc` in `ripserpp.cu`, wrap in RAII.
- [ ] Add per-`ripser` `cudaStream_t`; swap all `thrust::...` calls to
  `thrust::cuda::par.on(stream)`; add `stream` to raw `<<<..., stream>>>`.
- [ ] Add `mpcf::ripserpp::oom_error` (distinct from `cuda_error`) so
  the dispatcher can catch OOM specifically.
- [ ] Benchmark: re-run `scratch/bench_ph_scaling.py` variant with
  GPU on/off, measure wall-time and CPU/GPU utilization.

### Phase 4 — polish

- [ ] Docs: `docs/persistence.rst` — document `device=` or auto-detection,
  tradeoffs.
- [ ] `masspcf.system` knob: `limit_gpu_concurrency(n)` for the hybrid case.
- [ ] Coverage / edge cases: empty point clouds, n=1, distance-matrix input.
- [ ] Decide whether `RipserDistMatTask` also gets a GPU backend
  (sparse-distance-matrix specialization in Ripser++ exists). Point-clouds-
  only is the minimum viable first slice.

## Benchmark context to preserve

- Test file: `samples_3_50_2500.mpcf` (PointCloudTensor shape (3, 50), 150
  items of 2500 points, pcloud64). Loaded via `masspcf.io.load`.
- Benchmark script: `scratch/bench_ph_scaling.py` (scales worker count).
- Build: `cmake-build-release` is the perf-testing dir; `cmake-build-debug`
  for correctness. `cmake --install cmake-build-release` symlinks extensions
  into `masspcf/`. Per CLAUDE.md, run pytest from `test/` to avoid shadowing.

## Open questions / defer

- Should `max_dim > 1` route to GPU too? Upstream Ripser++ supports it but
  memory blows up fast. Maybe cap GPU path at max_dim=2 and force-fall-back
  higher dims to CPU.
- NUMA: on multi-socket hosts the hybrid dispatcher probably wants CPU
  workers pinned to the socket the GPU sits on. Deferred.
- Multi-GPU: out of scope for first cut. Executor's `cuda()` pool already
  exists but Ripser++ assumes `cudaSetDevice(0)` — audit and parameterize.

## Files touched so far (uncommitted)

```
3rd/ripserpp/LICENSE                                   (new)
include/mpcf/persistence/ripserpp/parallel_hashmap/*   (new, from upstream)
include/mpcf/persistence/ripserpp/sparsehash/*         (new, from upstream)
include/mpcf/persistence/ripserpp/phmap_interface.hpp  (new, header-only shim)
src/cuda/ripserpp.cu                                   (new, unpatched copy of upstream ripser++.cu)
```

No CMake hookup yet — the new `.cu` is not yet compiled.
