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

## Thread-safety issues — resolved

All items here were blockers for running more than one Ripser++ invocation
simultaneously on the GPU. Kept for the archaeology of why the code is
shaped the way it is.

1. **`list_of_barcodes` global** — resolved in Phase 2. Threaded through
   the `ripser` constructor as a `std::vector<std::vector<birth_death_coordinate>>&`
   output reference; the facade allocates a local vector and copies out.

2. **`phmap_interface` global hash map** — resolved in Phase 2. Now a
   per-instance `phmap::parallel_flat_hash_map<int64_t, int64_t> pivot_map`
   member; `phmap_put` / `phmap_get_value` / `phmap_clear` are member
   functions. The `phmap_interface.hpp` shim was deleted.

3. **Default CUDA stream** — resolved in Phase 3 (B). Each `ripser` instance
   owns a `cudaStream_t stream_` created in the ctor and destroyed in the
   dtor. Every kernel launch, thrust call (`thrust::cuda::par.on(stream_)`),
   memcpy/memset, and sync goes on that stream.

4. **Raw `cudaMalloc` without RAII** — resolved in Phase 3 (A). 16 device
   pointers + the local `d_num` became `mpcf::CudaDeviceArray<T>` members;
   the 24 cudaMalloc sites became `.allocate(n)`; cudaFrees and both
   `free_gpumem_*` methods were deleted. OOM mid-compute no longer leaks.

5. **`exit(...)` calls** — resolved in Phase 1. All replaced with
   `throw std::runtime_error(...)`; `main/run_main*/print_usage_and_exit`
   deleted. The CUDA error path throws `mpcf::cuda_error` (in
   `mpcf/cuda/cuda_util.cuh`), whose `code()` can be inspected for
   `cudaErrorMemoryAllocation` by the future dispatcher.

6. **OpenMP pragmas** — resolved in Phase 2. All 5 `#pragma omp parallel for
   schedule(guided,1)` sites became `tf::Taskflow::for_each_index` with
   `tf::GuidedPartitioner<>(1)`, running on an `mpcf::Executor&` threaded
   through the `ripser` constructor.

## Step-by-step plan

### Phase 1 — verbatim lift, make it build [DONE]

- [x] `3rd/ripserpp/LICENSE`
- [x] Vendored header-only deps moved to `include/mpcf/internal/{parallel_hashmap,sparsehash}/`
  (their cross-includes use top-level `<sparsehash/...>` / `<parallel_hashmap/...>`,
  so the `include/mpcf/internal/` dir is on `mpcf_cuda`'s PRIVATE include path).
- [x] `src/cuda/ripserpp.cu`: patched includes, `CUDACHECK` replaced with the
  project-wide `CHK_CUDA` (throws `mpcf::cuda_error` from
  `mpcf/cuda/cuda_util.cuh`), `main/run_main/run_main_filename/
  print_usage_and_exit` deleted, all `exit(...)` replaced with throws,
  `ripser_plusplus_result` C-API structs deleted.
- [x] CMake: `src/cuda/ripserpp.cu` added to the `mpcf_cuda` static library
  under `BUILD_WITH_CUDA`.
- [x] Facade `include/mpcf/persistence/ripserpp/ripserpp.hpp` declares
  `mpcf::ph::ripserpp::compute_barcodes_pcloud<T>(points, maxDim, out, exec)`;
  implementation is at the bottom of `src/cuda/ripserpp.cu`.

### Phase 2 — first functional use, single GPU job [DONE]

- [x] Thread `list_of_barcodes` through `compute_barcodes` as an instance
  member (kills global #1).
- [x] Replace OMP pragmas with taskflow `for_each_index` via a new
  `mpcf::Executor&` constructor argument on `ripser`.
- [x] Make the phmap_interface singleton an instance member of `ripser`
  (kills global #2). `phmap_interface.hpp` shim deleted.
- [x] Add `RipserPlusPlusTask<T>` in
  `include/mpcf/persistence/compute_persistence.hpp`, mirroring
  `RipserTaskImpl` but iterating items **serially** on the GPU for now.
- [x] pybind: `spawn_ripser_plusplus_pcloud_euclidean_task` gated on
  `BUILD_WITH_CUDA` so the symbol only exists in `_mpcf_cudaXX`.
- [x] Python: `compute_persistent_homology(..., device="cpu"|"gpu"|"auto")`.
- [x] Correctness: `test_ripser_plusplus.py` (8 tests) compares GPU vs
  CPU on rectangles, random, circle, tensor-of-pclouds, with
  tolerance via extended `Barcode::is_isomorphic_to(..., atol, rtol)`.

### Phase 3 — hybrid dispatcher + concurrent GPU jobs [DONE]

A, B, C, and D all landed. Remaining items captured in the
"Ship-minimum punch list" at the bottom of this doc.

- [x] (B) Per-`ripser` `cudaStream_t` threaded through every kernel launch,
  thrust call (`thrust::cuda::par.on(stream_)`), cudaMemcpy/cudaMemset,
  and sync. Ripser++ no longer uses the default stream. (Commit:
  `Put all Ripser++ GPU work on a per-instance CUDA stream`.)
- [x] (A) Raw `cudaMalloc` replaced with `mpcf::CudaDeviceArray<T>` members.
  The RAII class gained a public `allocate(sz)` / `reset()`, implicit
  `operator T*()`, `operator->()`, and `address_of_ptr()`. Upstream's
  `n >= 10` guard in `free_gpumem_*` (a workaround for a double-free bug
  in the repeated-single-point case) is fully removed — RAII fixes the
  underlying issue. (Commit: `RAII-wrap all Ripser++ cudaMalloc sites`.)
- [x] (C) **Hybrid dispatcher** via new `mpcf::GpuMemoryScheduler` in
  `include/mpcf/cuda/gpu_memory_scheduler.hpp` plus rewired
  `RipserPlusPlusTask`. Highlights:
  - Decoupled scheduler: problem-agnostic First-Fit online bin packing
    (Johnson 1974) across per-GPU memory budgets, with AIMD cost-factor
    calibration on OOM (Chiu & Jain 1989). Caller passes "cost units"
    (simplex count for Ripser++); scheduler owns per-GPU K (bytes per
    unit), budget, and the RAII `Reservation` that cudaSetDevice's on
    acquire and releases bytes on scope exit.
  - Baseline K for Ripser++ dense path = 64 bytes per simplex (sum of
    7 max_num_simplices-sized arrays: diameter_index_t_struct +
    value_t + index_t + value_t + index_t + index_t + index_t_pair_struct).
  - `RipserPlusPlusTask` dispatches via `parallel_walk_async` on
    `exec.cpu()`; per-item callback tries to reserve a GPU slot
    (cost=n*(n-1)/2), runs GPU on success, CPU Ripser on failure. On
    `cuda_error(cudaErrorMemoryAllocation)`, bumps K for that GPU via
    `record_oom()` and retries the item on CPU.
  - Why not `exec.cuda()`? That pool has num_gpus threads (1:1
    thread-to-GPU), which caps M = num_gpus. We want M > num_gpus
    (the whole purpose of per-instance streams); CPU workers picking
    GPU slots via the scheduler lets M scale with GPU memory.
  - Commit: `Add GpuMemoryScheduler and wire into RipserPlusPlusTask`.

Phase 3 landed:

- [x] **C-tests**: `test/test_gpu_memory_scheduler.cu` (commit
  `Add GpuMemoryScheduler tests, concurrency cap, and internals doc`).
  Also added `limit_gpu_concurrency(n)` Python knob and the
  `docs/internals/gpu_memory_scheduler.rst` internals doc.
- [x] (D) **Benchmark**. `benchmarks/bench_ph_hybrid.py` (cpu/gpu/auto
  with nvidia-smi sampling) and `benchmarks/profile_real.py` (NVTX-scoped
  driver for .mpcf inputs). Outputs in `benchmarks/bench_results/`.

Post-C perf / diagnostics (not on the original Phase 3 list, but landed
on this branch):

- Hybrid cost-estimator fix + queue-on-busy + upstream diagnostics
  (`upstream_cpu_fallback`, `gpu_max_dim` surfaced from ripser++).
- Serialize inner parallel-fors when batch > 1 so parallelism comes
  from concurrent ripser instances instead of subdividing each one.
- Self-calibrating K from first-admit memory snapshot (no more
  hand-tuned 64 bytes/simplex baseline).
- Swapped phmap → google dense_hash_map for the pivot map.
- GPU distance-matrix construction (eliminates a big H2D).
- `cudaMallocAsync` pool for ripser++ allocs and thrust temp storage.
- Event-driven `GpuMemoryScheduler::wait_for_reserve` (no polling).
- Dim-0 union-find short-circuits once the spanning forest closes.
- NVTX range around the timed PH call for nsys filtering.

### Ship-minimum punch list

Agreed scope for shipping this branch (everything else deferred):

- [ ] Multi-GPU correctness test, gated on `cudaGetDeviceCount() >= 2`
  (skips on single-GPU / CPU-only boxes). Asserts GPU barcodes match
  CPU reference. Lives in `test/python/persistence/test_ripser_plusplus.py`.
- [ ] Live-OOM Python test. Cloud large enough to actually OOM the
  device (~60k points on a 12 GB card); asserts the automatic CPU
  fallback still produces correct barcodes.
- [ ] Dense distance-matrix GPU path via new `RipserPlusPlusDistMatTask<T>`.
  The `ripser<compressed_lower_distance_matrix>` specialization already
  exists in `src/cuda/ripserpp.cu`; work is facade +
  `compute_barcodes_distmat<T>` + pybind + Python routing in
  `masspcf/persistence/homology.py`. **Sparse not in scope** — neither
  CPU nor GPU Ripser currently supports sparse distance matrices.
- [ ] `docs/persistence.rst`: document `device=`, GPU→CPU auto-fallback,
  `limit_gpus()` / `limit_gpu_concurrency()` knobs, and remove the
  "distmat is CPU-only" caveat once the dense distmat GPU path lands.

Explicitly deferred post-ship:

- NUMA pinning on multi-socket hosts (perf, not correctness).
- `max_dim > 1` GPU routing cap (upstream handles it; memory blows up
  fast — revisit if users hit it).
- Sparse distance-matrix Ripser++ kernels.

## Benchmark context to preserve

- Test file: `samples_3_50_2500.mpcf` (PointCloudTensor shape (3, 50), 150
  items of 2500 points, pcloud64). Loaded via `masspcf.io.load`.
- Benchmark script: `scratch/bench_ph_scaling.py` (scales worker count).
- Build: `cmake-build-release` is the perf-testing dir; `cmake-build-debug`
  for correctness. `cmake --install cmake-build-release` symlinks extensions
  into `masspcf/`. Per CLAUDE.md, run pytest from `test/` to avoid shadowing.

## Open questions / defer

- Should `max_dim > 1` route to GPU too? Upstream Ripser++ supports it but
  memory blows up fast. Deferred — users hitting the limit can
  `device="cpu"` for now.
- Multi-GPU: the scheduler does call `cudaSetDevice` on reservation and
  ripserpp.cu doesn't override it, so structurally it should work. Needs
  the gated multi-GPU correctness test (ship-minimum item #1) to actually
  verify.

## Current state (for picking up next session)

### Key files

- `src/cuda/ripserpp.cu` — patched port. No globals, no OMP, per-instance
  stream + RAII buffers, cudaMallocAsync pool, GPU-built distance matrix.
  The `mpcf::ph::ripserpp::compute_barcodes_pcloud` facade lives at the
  bottom (needs a sibling `compute_barcodes_distmat` for the dense
  distmat GPU path — ship-minimum item #3).
- `include/mpcf/persistence/ripserpp/ripserpp.hpp` — facade declaration
  with `Diagnostics` struct.
- `include/mpcf/persistence/compute_persistence.hpp` —
  `RipserPlusPlusTask<T>` hybrid dispatcher. A new
  `RipserPlusPlusDistMatTask<T>` goes here for the distmat GPU path.
- `include/mpcf/cuda/gpu_memory_scheduler.hpp` — First-Fit bin-packing
  scheduler with AIMD K calibration and self-calibration from first-admit
  snapshot. Event-driven wait; `limit_concurrency(n)` knob.
- `masspcf/persistence/homology.py` — `compute_persistent_homology(...,
  device="cpu"|"gpu"|"auto")`. Distmat branch currently ignores `device=`
  — remove that limitation as part of ship-minimum item #3.
- `masspcf/system.py` — `limit_gpus(n)`, `limit_gpu_concurrency(n)`.
- `test/python/persistence/test_ripser_plusplus.py` — correctness tests
  (add multi-GPU gated test + live-OOM test here).
- `test/test_gpu_memory_scheduler.cu` — gtest for scheduler internals.
- `benchmarks/bench_ph_hybrid.py`, `benchmarks/profile_real.py` — the
  hybrid-path benchmark and NVTX-scoped profiling driver.
