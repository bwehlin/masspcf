================================
GPU memory scheduler
================================

``mpcf::GpuMemoryScheduler`` (in ``include/mpcf/cuda/gpu_memory_scheduler.hpp``) is a problem-agnostic admission controller that decides when, and on which GPU, a piece of work is allowed to start. It is the mechanism that lets multiple Ripser++ instances coexist on the same GPU without OOM-thrashing, while still keeping every GPU saturated when work is plentiful.

The scheduler is independent of any particular workload. Callers describe the size of an item in abstract **cost units** (for Ripser++, the dominant per-dimension simplex count :math:`\binom{n}{\min(d_{\max}+1, \lfloor n/2 \rfloor)}` — equal to the edge count :math:`\binom{n}{2}` at ``max_dim = 1`` and growing faster for higher dimensions); the scheduler converts that to a byte estimate using a per-GPU cost factor :math:`K` (bytes per unit) that it auto-tunes from observed OOM events.


Problem
========

The hybrid persistence dispatcher (``RipserPlusPlusTask`` in ``include/mpcf/persistence/compute_persistence.hpp``) wants to:

1. Run **more than one** Ripser++ instance on the GPU concurrently. Each ``ripser`` instance (in ``src/cuda/ripserpp_impl.inc``) owns its own ``mpcf::CudaStream`` and routes all of its allocations (including thrust temp storage via ``mpcf::CudaAsyncMemoryResource``) through ``cudaMallocAsync`` on that stream, so the GPU happily overlaps multiple instances — but only if each one's working set fits in device memory at the same time.
2. **Fall back to CPU** for items that cannot fit, instead of failing the whole batch.
3. Make this decision **without pre-sorting** the input tensor (which may be very large) and **without assuming uniform item size**.
4. Generalize to **multiple GPUs**.

A naive "max N concurrent ripsers" knob fails on (3) and (4): the right N depends on item size and per-GPU free memory, and both vary at runtime.


Algorithm
==========

The scheduler is a thin wrapper around online **First-Fit bin packing** :footcite:`Johnson1974`, with two additions: items are sized by a learned cost factor :math:`K` rather than a fixed size, and overshoots are corrected by an **AIMD** :footcite:`ChiuJain1989` rule on :math:`K`.


Per-GPU state
--------------

For each visible GPU :math:`i` the scheduler holds:

- ``budget`` — the bin capacity, a fraction of free memory at construction (``budget_fraction``, default 0.6). The remainder absorbs CUDA scratch, fragmentation, and other tenants.
- ``remaining`` — bytes still available, an ``std::atomic<int64_t>``.
- ``k_bytes`` — current cost factor :math:`K_i` (bytes per cost unit), an ``std::atomic<double>``.
- ``calibrated`` — ``std::atomic<bool>``, set once the first-admit calibration measurement on this GPU has completed (see "First-admit calibration" below). Further admissions skip that measurement.
- ``activity_gen`` — ``std::atomic<uint64_t>`` admission counter, bumped on every successful ``try_reserve``. Used by the calibration-window contamination guard (see "First-admit calibration" below).

Initial :math:`K` comes from the caller (``initial_k_bytes_per_unit``) and reflects an audited or measured baseline for the workload. For Ripser++ on the dense path, the baseline is ~64 bytes per simplex (sum of seven ``max_num_simplices``-sized arrays — diameter struct, two value arrays, three index arrays, and an index pair struct).


Optional concurrency cap
-------------------------

In addition to the per-GPU memory budget, the scheduler accepts an optional ``Config::max_concurrent`` cap on the **total** number of active reservations across all GPUs. The default ``0`` means no cap (memory is the only limit). When set, ``try_reserve`` short-circuits as soon as ``cap`` reservations are already live, returning an inactive Reservation without walking GPUs.

The cap is exposed to users as :func:`masspcf.system.limit_gpu_concurrency`. Its main use cases are benchmarking (sweeping M to find the cooperation sweet spot — see ``benchmarks/bench_ph_hybrid.py``) and operating in shared environments where the user wants to leave GPU headroom for other tenants.

Enforcement uses the same atomic-rollback idiom as the byte budget: a tentative ``fetch_add`` on the active counter, rolled back if no GPU has room. The cap and the per-GPU bytes are independent; either limit can deny admission.


Reservations: active vs. inactive
----------------------------------

The unit of admission is a ``Reservation`` — a small, move-only RAII handle returned by ``try_reserve``. ``Reservation::active()`` returns ``true`` iff this object currently owns a granted GPU slot — meaning ``try_reserve`` succeeded for it, ``cudaSetDevice(gpu_index())`` has been called on the calling thread, and ``bytes()`` worth of budget has been subtracted from ``remaining`` on that GPU. The destructor of an active Reservation returns those bytes.

A Reservation is **inactive** in three cases, all distinguished internally by ``m_sched == nullptr``:

1. **Default-constructed** (``Reservation{}``) — never held a slot.
2. **Returned from a failed** ``try_reserve`` — either ``cost_units <= 0``, or no GPU had room. No ``cudaSetDevice`` was called and no bytes are owed.
3. **Moved-from** — ownership transferred elsewhere; this object is now empty.

An inactive Reservation is safe to destroy (no-op release) and is the signal the caller uses to fall back to CPU.


Reserve
--------

``try_reserve(cost_units)`` walks GPUs in order :math:`0, 1, \dots, N-1` (First-Fit) and tries to admit the item:

.. code-block:: text

   for each gpu i:
     est_i = K_i * cost_units
     if est_i > budget_i: continue           # structurally too big for this bin
     prev = remaining_i.fetch_sub(est_i)     # tentatively admit
     if prev >= est_i:                       # really fit
       cudaSetDevice(i)
       return Reservation(this, i, est_i)
     remaining_i.fetch_add(est_i)            # roll back, try next gpu
   return inactive Reservation

The ``fetch_sub``/``fetch_add`` pair is the standard atomic "try to take, roll back on failure" idiom; it lets the check be lock-free and tolerates many concurrent admissions racing for the last few bytes.

First-Fit is a 17/10-competitive online algorithm under the standard adversarial model :footcite:`Johnson1974`. We do not need anything stronger: items are not adversarial (sizes come from the user's data, not an opponent), and we do not pay the cost of sorting the input or buffering items.


Release
--------

Successful ``try_reserve`` returns an RAII ``Reservation``. Its destructor calls ``release_bytes(gpu_idx, bytes)``, which is a single ``remaining.fetch_add(bytes)``. The Reservation is move-only and idempotent on release; assigning over a live Reservation releases the prior one first.

This means a caller writes:

.. code-block:: cpp

   if (auto r = sched.try_reserve(cost_units); r.active()) {
     run_on_gpu(item, r.gpu_index());
   } else {
     run_on_cpu(item);
   }
   // r.~Reservation() returns the bytes


Wait variant
-------------

``wait_for_reserve(cost_units, max_wait)`` is a blocking companion to ``try_reserve``. It first runs a structural-fit check -- :math:`K_i \cdot \text{cost\_units} \leq \text{budget}_i` for at least one GPU -- and returns an inactive Reservation immediately if no device can *ever* hold the item under current :math:`K`. Since :math:`K` is monotone non-decreasing (AIMD only bumps it up on OOM), waiting in that case would be pointless.

Otherwise it loops: ``try_reserve``; on failure, acquire ``m_wait_mutex`` and re-try under the lock (to close the race with a release that completed between the first try and the lock acquire); if still no slot, block on ``m_wait_cv`` until a peer's ``release_bytes`` or ``record_oom`` fires ``notify_all``. Both sides fence through the same mutex, so no wakeup is lost and no polling is required.

The default ``max_wait`` is ``std::chrono::steady_clock::duration::max()`` (unbounded). Finite timeouts return an inactive Reservation on expiry, with one last ``try_reserve`` to catch a notify that arrived in the timeout window.

``wait_for_reserve`` is meant for workloads where the GPU is much faster per item than the CPU fallback -- queueing then beats running anything on CPU. The hybrid Ripser++ dispatcher picks between ``try_reserve`` and ``wait_for_reserve`` based on the user-settable ``mpcf::settings().hybridGpuQueueOnBusy`` (exposed to Python as ``cpp.set_hybrid_gpu_queue_on_busy``). OOM-triggered CPU fallback remains unconditional regardless of this setting: the AIMD bump is already the right signal that the item will not fit.


First-admit calibration
------------------------

Before any OOM happens, the scheduler tries to *measure* rather than guess the real :math:`K`. The first admitted item on each GPU is also a calibration run: ``try_reserve`` snapshots ``cudaMemGetInfo(free_before)`` at admit, stores it on the ``Reservation``, and ``release_bytes`` snapshots ``free_after`` on release. The observation is:

.. math:: K_{\text{obs}} = (\text{free\_before} - \text{free\_after}) / \text{cost\_units}

which is passed to ``note_observed_k``. That call raises :math:`K` monotonically: ``K := max(K, K_obs)``. It never lowers it, matching the AIMD "trust observed > modeled, but not the reverse" policy. After the measurement the per-GPU ``calibrated`` flag is set; subsequent admissions skip the snapshot but keep reacting to OOMs.

The measurement is taken on the **live-device** constructor only. The test-injected constructor leaves ``m_calibrate_enabled = false``, so unit tests drive :math:`K` directly via ``note_observed_k`` rather than reading CUDA.

**Why we need it.** ``initial_k_bytes_per_unit`` is an audit of *declared* per-simplex storage (64 bytes for Ripser++ dense), but the *real* residency also includes scratch buffers, thrust temp storage, and upstream's memory planner's slack. Waiting for an OOM to learn that delta is wasteful when we can measure it cheaply on the very first item.

Contamination guard
^^^^^^^^^^^^^^^^^^^^

The release-time snapshot is only meaningful if the window was exclusive: if a peer reservation admitted during our window, its peak residency would show up in ``free_after`` too and inflate :math:`K_{\text{obs}}`. Checking ``m_active == 1`` at release is necessary but not sufficient — the CUDA memory pool's release threshold is set to ``UINT64_MAX`` (to keep allocations pool-local and fast), so a peer that admitted, allocated, and released entirely inside our window leaves its bytes cached in the pool. ``cudaMemGetInfo`` still reports those bytes as "used", even though the peer is no longer live.

The fix is the per-GPU ``activity_gen`` counter. ``try_reserve`` does ``g.activity_gen.fetch_add(1)`` after a successful admit and stores the post-increment value on the ``Reservation``. ``release_bytes`` re-reads the counter and only takes the measurement if it matches — i.e. no peer admitted in between. The ``calibrated`` flag is set unconditionally either way, so a contaminated window just means we fall back to AIMD on OOM rather than pre-empt it.


OOM and AIMD
-------------

Even with a careful initial :math:`K`, the actual peak residency of a real workload varies with the data — fragmentation, transient working sets, and instance-to-instance overlap can push us past the estimate. When that happens, the GPU code throws ``mpcf::cuda_error`` (defined in ``include/mpcf/cuda/cuda_util.cuh``) with ``code() == cudaErrorMemoryAllocation``. The dispatcher catches it and calls ``record_oom(gpu_idx)``:

.. code-block:: text

   K_i := K_i * oom_backoff           # oom_backoff defaults to 1.5

This is the **multiplicative-decrease** half of an AIMD controller :footcite:`ChiuJain1989`. The next ``try_reserve`` on that GPU will book ``oom_backoff`` times more bytes per cost unit, so fewer items fit concurrently and the chance of another OOM drops.

The "additive increase" half is implicit: every successful run with the higher :math:`K` leaves headroom unused, which is what we want — the system has revealed itself to be tighter than we modeled, and we choose to leave the slack alone rather than chase it back. (A full AIMD that ratchets :math:`K` back down on long stretches of success is plausible future work, but our items have substantial run-to-run variance and we prefer the conservative steady state.)

The CAS loop in ``record_oom`` (``compare_exchange_weak`` on ``k_bytes``) makes concurrent OOMs safe: if two instances OOM in close succession, both bumps land and :math:`K` reaches the correct multiplier.


Multi-GPU
==========

The live constructor calls ``cudaGetDeviceCount`` and creates one ``GpuState`` per device, each with its own budget, remaining, and :math:`K`. ``try_reserve`` walks them in index order (First-Fit). When it finds a fit, it calls ``cudaSetDevice`` on the calling thread before returning the Reservation; the caller can then issue CUDA work without further device juggling.

This is why CPU workers — not the executor's CUDA pool — are the ones picking GPU slots. The CUDA pool has one thread per GPU (1:1), which would cap concurrency at ``num_gpus``. Letting CPU workers acquire reservations and call ``cudaSetDevice`` themselves lets the number of concurrent GPU instances scale with **memory** rather than thread count.

Per-GPU :math:`K` means OOM on one GPU does not penalize another. This matters when a heterogeneous host has a small consumer card and a big accelerator: the small card's :math:`K` will climb, the big card's will not, and the steady-state distribution of work follows.


RAII reservation
=================

The ``Reservation`` class is move-only and small (scheduler pointer, gpu index, byte count, plus the calibration-window fields: ``cost_units``, ``free_before``, a ``calibrate`` flag, and ``admit_gen``). ``release()`` is ``noexcept`` and idempotent. The destructor returns the bytes to the per-GPU ``remaining`` counter — there is nothing else to clean up, since the Reservation does not own any CUDA resources itself; it only owns the right to allocate them.

Two implications:

- **Exception safety is automatic.** If the workload throws after the reservation is granted (including the ``mpcf::cuda_error`` that triggers OOM fallback), the Reservation's destructor still runs and the budget is returned. The bytes are not leaked even though the actual cudaMalloc may have failed.
- **The reservation does not guarantee a successful cudaMalloc.** It is an *admission* decision, not an allocation. The actual ``cudaMalloc`` happens later (inside the workload) and can still fail; that is exactly what triggers the AIMD bump.


Test injection
===============

There are two constructors:

.. code-block:: cpp

   explicit GpuMemoryScheduler(Config cfg);
   GpuMemoryScheduler(std::vector<std::int64_t> per_device_budgets, Config cfg);

The second bypasses ``cudaMemGetInfo`` entirely and lets a unit test pin per-device budgets to known values. This is what ``test/test_gpu_memory_scheduler.cu`` uses to cover the algorithm independently of host topology — small budgets so a handful of reservations is enough to exhaust the bin, and synthetic multi-GPU configurations even on single-GPU CI hosts.

Both constructors initialize :math:`K` from ``Config::initial_k_bytes_per_unit``.


Instrumentation
================

Each scheduler maintains cumulative atomic counters that partition every admission attempt into exactly one bucket:

- ``total_admitted`` — ``try_reserve`` / ``wait_for_reserve`` succeeded.
- ``total_failed_no_room`` — denied because no GPU had enough bytes remaining.
- ``total_failed_cap`` — denied because ``max_concurrent`` was hit (short-circuited before walking GPUs).
- ``total_oom`` — ``record_oom`` was called (a hard or soft OOM observed by the caller).
- ``peak_active`` — the high-water mark of simultaneously live reservations.

These live on the scheduler and are readable via ``total_admitted()``, ``total_failed_no_room()``, ``total_failed_cap()``, ``total_oom()``, and ``peak_active()``. Separating cap-denials from memory-denials is useful when tuning ``gpuConcurrencyCap``: a run dominated by ``total_failed_cap`` means the cap is the binding constraint, while one dominated by ``total_failed_no_room`` means memory is.

On destruction, the scheduler snapshots all of the above into the global ``mpcf::LastSchedulerStats`` under ``last_gpu_scheduler_stats_mutex()``. Python reads the snapshot via ``cpp.get_last_gpu_scheduler_stats()`` — the hybrid PH benchmark harness (``benchmarks/bench_ph_hybrid.py``) uses this to sweep ``gpuConcurrencyCap`` / ``gpuBudgetFraction`` without poking at scheduler internals during the run.


Integration with RipserPlusPlusTask
====================================

In ``RipserPlusPlusTask`` / ``RipserPlusPlusDistMatTask`` (``include/mpcf/persistence/compute_persistence.hpp``):

- The task constructs a ``GpuMemoryScheduler`` once, seeding ``initial_k_bytes_per_unit`` with the audited :math:`K_0 = 64` bytes per simplex and wiring ``budget_fraction`` / ``max_concurrent`` to the user-tunable ``mpcf::settings().gpuBudgetFraction`` and ``mpcf::settings().gpuConcurrencyCap`` (exposed to Python as ``cpp.set_gpu_budget_fraction`` and ``cpp.limit_gpu_concurrency`` / :func:`masspcf.system.limit_gpu_concurrency`).
- ``parallel_walk_async`` runs N CPU worker threads over the input point-cloud or distance-matrix tensor. For each item of size :math:`n`, the dispatcher computes ``cost_units = simplex_cost_units(n, max_dim)`` = :math:`\binom{n}{\min(d_{\max}+1,\, \lfloor n/2 \rfloor)}`. This matches the dominant-term cap used by the upstream memory planner's ``max_num_simplices_forall_dims`` (see ``calculate_gpu_dim_max_for_fullrips_computation_from_memory`` in ``ripserpp_impl.inc``): it collapses to :math:`\binom{n}{2} = n(n-1)/2` for ``max_dim = 1`` and grows faster for higher ``max_dim``. The computation is saturated at ``INT64_MAX`` so huge items are treated as "never fits" rather than overflowing.
- Admission policy is selected by ``mpcf::settings().hybridGpuQueueOnBusy``: ``try_reserve`` (default -- immediate CPU fallback if busy) or ``wait_for_reserve`` (queue until a slot frees). Either returns an inactive Reservation if no GPU can structurally fit the item at current :math:`K`; the dispatcher then runs the CPU Ripser path for that item.
- On success: run Ripser++ on the chosen GPU, on a fresh per-instance ``mpcf::CudaStream``. The ripser++ facade (``compute_barcodes_pcloud`` / ``compute_barcodes_distmat`` in ``include/mpcf/persistence/ripserpp/ripserpp.hpp``) also takes a ``parallel_inner_loops`` flag -- the dispatcher sets it ``true`` only when the batch size is at most ~``hw_concurrency / 4``, so for larger batches the ripser++ internal parallel-for loops run serially on the calling worker and parallelism comes entirely from running many ripser instances concurrently rather than from subdividing each one.
- The facade returns per-invocation ``ripserpp::Diagnostics`` (``upstream_cpu_fallback``, ``gpu_max_dim``). A set ``upstream_cpu_fallback`` means ripser++'s own embedded memory planner lowered ``gpu_max_dim`` below the requested ``max_dim`` and ran the high-dimensional tail on CPU -- the barcodes are still correct, but our :math:`K` under-estimated the item. The dispatcher treats this as a **soft OOM** and calls ``record_oom(gpu_idx)``, so subsequent items of similar size over-book less aggressively even though no ``cudaErrorMemoryAllocation`` was thrown.
- On ``mpcf::cuda_error`` with ``cudaErrorMemoryAllocation`` mid-run (**hard OOM**): ``record_oom(gpu_idx)`` and re-run the item on CPU. Other ``cuda_error`` codes propagate. Thrust temp-storage OOMs are routed through ``mpcf::cuda_error`` too (see ``CudaAsyncMemoryResource::do_allocate`` in ``cuda_async_memory_resource.cuh``) so every OOM shape the ripser path can emit hits this catch uniformly.

The result: M concurrent GPU instances, where M is the largest number of items whose admitted byte estimates fit in the per-GPU budget -- naturally adapting to item size, free memory, and observed OOM history (hard and soft), with ``gpuBudgetFraction`` and ``gpuConcurrencyCap`` as the headroom/concurrency knobs and ``hybridGpuQueueOnBusy`` selecting queue-vs-fallback policy.


References
==========


.. footbibliography::
