================================
GPU memory scheduler
================================

``mpcf::GpuMemoryScheduler`` (in ``include/mpcf/cuda/gpu_memory_scheduler.hpp``) is a problem-agnostic admission controller that decides when, and on which GPU, a piece of work is allowed to start. It is the mechanism that lets multiple Ripser++ instances coexist on the same GPU without OOM-thrashing, while still keeping every GPU saturated when work is plentiful.

The scheduler is independent of any particular workload. Callers describe the size of an item in abstract **cost units** (for Ripser++, "number of edges in the dense filtration"); the scheduler converts that to a byte estimate using a per-GPU cost factor :math:`K` (bytes per unit) that it auto-tunes from observed OOM events.


Problem
========

The hybrid persistence dispatcher (``RipserPlusPlusTask`` in ``include/mpcf/persistence/compute_persistence.hpp``) wants to:

1. Run **more than one** Ripser++ instance on the GPU concurrently. Each instance owns its own CUDA stream (see ``src/cuda/ripserpp.cu``), so the GPU happily overlaps multiple kernels — but only if each instance's working set fits in device memory at the same time.
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

Initial :math:`K` comes from the caller (``initial_k_bytes_per_unit``) and reflects an audited or measured baseline for the workload. For Ripser++ on the dense path, the baseline is ~64 bytes per simplex (sum of seven ``max_num_simplices``-sized arrays — diameter struct, two value arrays, three index arrays, and an index pair struct).


Optional concurrency cap
-------------------------

In addition to the per-GPU memory budget, the scheduler accepts an optional ``Config::max_concurrent`` cap on the **total** number of active reservations across all GPUs. The default ``0`` means no cap (memory is the only limit). When set, ``try_reserve`` short-circuits as soon as ``cap`` reservations are already live, returning an inactive Reservation without walking GPUs.

The cap is exposed to users as :func:`masspcf.system.limit_gpu_concurrency`. Its main use cases are benchmarking (sweeping M to find the cooperation sweet spot — see ``scratch/bench_ph_hybrid.py``) and operating in shared environments where the user wants to leave GPU headroom for other tenants.

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

   if (auto r = sched.try_reserve(edges); r.active()) {
     run_on_gpu(item, r.gpu_index());
   } else {
     run_on_cpu(item);
   }
   // r.~Reservation() returns the bytes


OOM and AIMD calibration
-------------------------

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

The ``Reservation`` class is move-only and trivially small (a scheduler pointer, a gpu index, a byte count). ``release()`` is ``noexcept`` and idempotent. The destructor returns the bytes to the per-GPU ``remaining`` counter — there is nothing else to clean up, since the Reservation does not own any CUDA resources itself; it only owns the right to allocate them.

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


Integration with RipserPlusPlusTask
====================================

In ``RipserPlusPlusTask`` (``include/mpcf/persistence/compute_persistence.hpp``):

- The task constructs a ``GpuMemoryScheduler`` once with the workload's baseline :math:`K_0 = 64` bytes per simplex.
- ``parallel_walk_async`` runs N CPU worker threads over the input point-cloud tensor. Each worker, for its assigned item, calls ``try_reserve(n*(n-1)/2)`` (the number of edges = ``max_num_simplices`` for ``max_dim=1``).
- On success: run Ripser++ on the chosen GPU, on a fresh per-instance stream.
- On inactive Reservation (no GPU has room): run the CPU Ripser path for that item.
- On ``mpcf::cuda_error`` with ``cudaErrorMemoryAllocation`` mid-run: ``record_oom(gpu_idx)`` and re-run the item on CPU. Other ``cuda_error`` codes propagate.

The result: M concurrent GPU instances, where M is the largest number of items whose admitted byte estimates fit in the per-GPU budget — naturally adapting to item size, free memory, and observed OOM history, with no caller-visible knob beyond ``budget_fraction``.


References
==========


.. footbibliography::
