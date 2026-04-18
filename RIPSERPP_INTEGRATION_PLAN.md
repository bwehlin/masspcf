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

### Phase 3 — hybrid dispatcher + concurrent GPU jobs

Progress so far: A, B, and C landed; D and test harness remain.

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

Still to do for Phase 3:

- [ ] **C-tests**: gtest for `GpuMemoryScheduler` in isolation
  (test-injected budgets via the second constructor, cover reserve /
  release / OOM / multi-GPU First-Fit / concurrent stress). A
  file draft is already sketched in conversation; write it to
  `test/test_gpu_memory_scheduler.cu` and add to `MPCF_TEST_SOURCES_*`
  under `BUILD_CUDA_TESTER`.
- [ ] **C-live-OOM**: force a large enough point cloud on a small GPU
  to trigger the OOM→CPU fallback path end-to-end. Today the fallback
  is only covered by the gtest's simulated OOM; we want at least one
  Python test that actually OOMs the device.
- [ ] (D) **Benchmark**. Extend `scratch/bench_ph_scaling.py` (or a
  new script) with a `device=` switch. Measure:
  - wall-time for the full batch with `device="cpu"`, `device="gpu"`,
    `device="auto"` (picks GPU when backend loaded);
  - per-GPU utilization (nvidia-smi sampled during the run) to
    confirm M > 1 concurrent ripsers actually happens for the
    2500-point workload;
  - memory high-water via `nvidia-smi --query-gpu=memory.used`.
  Inputs: `samples_3_50_2500.mpcf` (150 items of 2500 points) and a
  synthetic 10k-point batch (GPU shines here: from today's single-item
  timing, CPU ≈ 10.3 s for 2500 pts but GPU single-stream is ~85 s
  for 10k pts; batch scaling matters).

### Where to pick up next session (Phase 3 tests + D)

- `include/mpcf/cuda/gpu_memory_scheduler.hpp` is in place and used.
- `RipserPlusPlusTask::dispatch_item` in
  `include/mpcf/persistence/compute_persistence.hpp` is the hybrid
  loop — no changes pending.
- Next concrete step: write `test/test_gpu_memory_scheduler.cu`. The
  test plan (with expected cases) is in the earlier conversation
  turn; main cases are:
    NoDevicesGivesInactiveReservations, ZeroOrNegativeCostYieldsInactive,
    ReserveDeductsFromRemaining, DestructorReleasesBudget,
    OversizedItemReturnsInactive, FirstFitAcrossMultipleGpus,
    RunningOutOfEverythingGivesInactive, OomBumpsKForThatGpuOnly,
    RaisedKReducesAvailableSlotsOnThatGpu, ConcurrentReservationsAreAtomic.
  Add to `MPCF_TEST_SOURCES_CUDA` (under BUILD_CUDA_TESTER block in
  CMakeLists.txt around line 711).
- Then run the benchmark and record results back in this doc.

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

## Current state (as of pause — Phase 3 A+B done, C+D to go)

Everything through `git log --oneline ripserpp-integration` on this branch.
In rough order of landing:

1. `Vendor Ripser++ and start integration plan` — initial copy + plan doc.
2. `Wire Ripser++ through Phase 1` — makes it build, no functional use.
3. `ignore local artifacts` — housekeeping.
4. `Route GPU Ripser++ through compute_persistent_homology` — Phase 2 MVP:
   facade + `RipserPlusPlusTask` + pybind + Python `device=` + 8
   correctness tests. `Barcode::is_isomorphic_to` gained `atol`/`rtol`.
5. `Remove Ripser++ file-scope globals and OpenMP dependency` — kills
   `list_of_barcodes` + phmap singleton; `#pragma omp` → taskflow
   `for_each_index` on an injected `mpcf::Executor&`.
6. `Put all Ripser++ GPU work on a per-instance CUDA stream` — Phase 3 B.
7. `RAII-wrap all Ripser++ cudaMalloc sites with CudaDeviceArray` — Phase 3 A.
8. `Add GpuMemoryScheduler and wire into RipserPlusPlusTask` — Phase 3 C
   (dispatcher landed; gtest and bench still to do).

### Key files

- `src/cuda/ripserpp.cu` — patched port. No globals, no OMP, per-instance
  stream + RAII buffers. The `mpcf::ph::ripserpp::compute_barcodes_pcloud`
  facade lives at the bottom.
- `include/mpcf/persistence/ripserpp/ripserpp.hpp` — facade declaration.
- `include/mpcf/persistence/compute_persistence.hpp` — `RipserPlusPlusTask<T>`
  (CUDA-gated) that iterates items **serially**. Needs to become parallel
  for Phase 3 C.
- `include/mpcf/cuda/cuda_util.cuh` — `mpcf::cuda_error` thrown by
  `CHK_CUDA`. The dispatcher will `catch` this and check `code() ==
  cudaErrorMemoryAllocation`.
- `include/mpcf/cuda/cuda_device_array.cuh` — extended with public
  `allocate`, `reset`, `operator T*()`, `operator->()`, `address_of_ptr()`.
- `include/mpcf/internal/{parallel_hashmap,sparsehash}/` — vendored
  header-only deps.
- `masspcf/persistence/homology.py` — `compute_persistent_homology(...,
  device="cpu"|"gpu"|"auto")`.
- `test/python/persistence/test_ripser_plusplus.py` — 8 correctness tests.

### Where to pick up (Phase 3 C)

`include/mpcf/persistence/compute_persistence.hpp` around the
`RipserPlusPlusTask<T>::run_async` method. Currently:

```cpp
tf::Taskflow flow;
flow.emplace([this]() {
  walk(m_input, [this](const std::vector<size_t>& index) {
    if (stop_requested()) return;
    process_item(index);  // always GPU path
    add_progress(1);
  });
});
return exec.cpu()->run(std::move(flow));
```

Target shape: N CPU workers + M GPU workers, shared atomic counter,
`try { process_item_gpu(i) } catch (mpcf::cuda_error& e) { if OOM
process_item_cpu(i) else rethrow }`. The CPU path can factor out of
`detail::compute_persistence_euclidean_single_impl`.
