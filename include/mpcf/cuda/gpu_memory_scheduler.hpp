/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef MPCF_GPU_MEMORY_SCHEDULER_H
#define MPCF_GPU_MEMORY_SCHEDULER_H

#include "cuda_util.cuh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace mpcf
{
  /// Snapshot of cumulative scheduler activity since the most recent
  /// GpuMemoryScheduler instance was destroyed. Populated automatically
  /// on every scheduler destruction; readable from Python via
  /// `cpp.get_last_gpu_scheduler_stats()` for benchmarking and
  /// diagnostics. Problem-agnostic: the scheduler itself doesn't know
  /// what callers did about failed reservations or OOMs, only that they
  /// happened.
  struct LastSchedulerStats
  {
    std::int64_t total_admitted = 0;
    std::int64_t total_failed_no_room = 0;  ///< rejected: no GPU had memory
    std::int64_t total_failed_cap = 0;      ///< rejected: concurrency cap hit
    std::int64_t total_oom = 0;
    int peak_active = 0;
    std::size_t num_gpus = 0;
  };

  inline LastSchedulerStats& last_gpu_scheduler_stats()
  {
    static LastSchedulerStats instance;
    return instance;
  }

  inline std::mutex& last_gpu_scheduler_stats_mutex()
  {
    static std::mutex m;
    return m;
  }

  /// Online First-Fit bin-packing scheduler over per-GPU memory budgets,
  /// with AIMD-style calibration of the per-GPU cost factor.
  ///
  /// Problem-agnostic: callers supply "cost units" (e.g., simplex count,
  /// matrix cells, rows) and an initial bytes-per-unit K. The scheduler
  /// owns an atomic `remaining` budget and the current K per device; it
  /// returns a RAII Reservation that has already called cudaSetDevice on
  /// the chosen GPU. On OOM, callers invoke record_oom(gpu_idx) which
  /// multiplicatively increases that GPU's K (so subsequent reservations
  /// over-book less aggressively); release happens automatically on
  /// Reservation destruction.
  ///
  /// References:
  ///   Johnson, D.S. (1974). Fast algorithms for bin packing.
  ///     J. Comput. System Sci. 8(3), 272-314.
  ///   Chiu, D.-M. and Jain, R. (1989). Analysis of the increase and
  ///     decrease algorithms for congestion avoidance in computer
  ///     networks. Computer Networks and ISDN Systems 17(1), 1-14.
  class GpuMemoryScheduler
  {
  public:
    struct Config
    {
      /// Fraction of free GPU memory (at construction) reserved as
      /// the scheduling budget. The remainder absorbs fragmentation,
      /// CUDA stream scratch, and other tenants.
      double budget_fraction = 0.6;

      /// Initial bytes-per-cost-unit estimate. Example: for a workload
      /// whose cost units are simplices and each concurrent instance
      /// residency is ~64 bytes per simplex, pass 64.0.
      double initial_k_bytes_per_unit = 1.0;

      /// Multiplicative backoff applied to the device's K on OOM.
      double oom_backoff = 1.5;

      /// Hard cap on the number of concurrent active reservations the
      /// scheduler will hand out across all GPUs. 0 means no cap; the
      /// scheduler is then bounded only by per-GPU memory budgets.
      /// When set, `try_reserve` returns an inactive Reservation as
      /// soon as `cap` reservations are live, without walking GPUs.
      int max_concurrent = 0;
    };

    /// RAII reservation. Holding one means: cudaSetDevice(gpu_index())
    /// has been called on this thread and `bytes()` bytes are subtracted
    /// from the chosen GPU's remaining budget. The destructor returns
    /// the bytes to the budget. Move-only.
    class Reservation
    {
    public:
      Reservation() = default;

      Reservation(const Reservation&) = delete;
      Reservation& operator=(const Reservation&) = delete;

      Reservation(Reservation&& o) noexcept
        : m_sched(o.m_sched), m_gpu(o.m_gpu), m_bytes(o.m_bytes),
          m_cost_units(o.m_cost_units), m_free_before(o.m_free_before),
          m_calibrate(o.m_calibrate)
      {
        o.m_sched = nullptr;
        o.m_gpu = -1;
        o.m_bytes = 0;
        o.m_cost_units = 0;
        o.m_free_before = 0;
        o.m_calibrate = false;
      }

      Reservation& operator=(Reservation&& o) noexcept
      {
        if (this != &o) {
          release();
          m_sched = o.m_sched;
          m_gpu = o.m_gpu;
          m_bytes = o.m_bytes;
          m_cost_units = o.m_cost_units;
          m_free_before = o.m_free_before;
          m_calibrate = o.m_calibrate;
          o.m_sched = nullptr;
          o.m_gpu = -1;
          o.m_bytes = 0;
          o.m_cost_units = 0;
          o.m_free_before = 0;
          o.m_calibrate = false;
        }
        return *this;
      }

      ~Reservation() { release(); }

      [[nodiscard]] bool active() const noexcept { return m_sched != nullptr; }
      [[nodiscard]] int gpu_index() const noexcept { return m_gpu; }
      [[nodiscard]] std::int64_t bytes() const noexcept { return m_bytes; }

    private:
      friend class GpuMemoryScheduler;
      Reservation(GpuMemoryScheduler* s, int gpu, std::int64_t bytes,
                  std::int64_t cost_units, std::size_t free_before, bool calibrate)
        : m_sched(s), m_gpu(gpu), m_bytes(bytes),
          m_cost_units(cost_units), m_free_before(free_before),
          m_calibrate(calibrate) {}

      void release() noexcept;

      GpuMemoryScheduler* m_sched = nullptr;
      int m_gpu = -1;
      std::int64_t m_bytes = 0;
      std::int64_t m_cost_units = 0;
      std::size_t m_free_before = 0;
      bool m_calibrate = false;
    };

    /// Construct using live CUDA device queries. One GpuState per device
    /// visible to cudaGetDeviceCount; budget per device is
    /// `budget_fraction * free_memory_at_construction`.
    explicit GpuMemoryScheduler(Config cfg);

    /// Test-friendly constructor: inject per-device budgets in bytes,
    /// bypassing cudaMemGetInfo. Used by unit tests and benchmarks that
    /// want deterministic capacity.
    GpuMemoryScheduler(std::vector<std::int64_t> per_device_budgets, Config cfg);

    GpuMemoryScheduler(const GpuMemoryScheduler&) = delete;
    GpuMemoryScheduler& operator=(const GpuMemoryScheduler&) = delete;

    /// On destruction, snapshot cumulative counters into the global
    /// `last_gpu_scheduler_stats()` for later read-back from Python.
    ~GpuMemoryScheduler();

    /// Probe GPUs in order 0..N-1 for a slot big enough for `cost_units`.
    /// On success: cudaSetDevice is called and a Reservation is returned.
    /// On failure (no GPU has enough room, or cost exceeds every GPU's
    /// total budget): an inactive Reservation is returned.
    Reservation try_reserve(std::int64_t cost_units);

    /// Like `try_reserve`, but blocks up to `max_wait` for a slot to
    /// become available if the item structurally fits on some GPU
    /// (i.e. `K * cost_units <= budget` for at least one device).
    /// Returns an inactive Reservation in three cases:
    ///   * the item cannot fit on any visible GPU under the current K
    ///     values (no point in waiting, since K is monotone non-decreasing),
    ///   * `max_wait` elapses before a slot becomes available,
    ///   * `cost_units <= 0`.
    /// The wait is strictly event-driven: `release_bytes` (on reservation
    /// release) and `record_oom` (on AIMD K bump) both synchronize via
    /// `m_wait_mutex` and `notify_all` so no wakeups are lost and no
    /// polling is needed.
    ///
    /// Pass `std::chrono::steady_clock::duration::max()` (the default)
    /// for an unbounded wait. Use this when the caller would rather
    /// queue than fall back to CPU -- e.g. when GPU is expected to be
    /// much faster than CPU per item.
    Reservation wait_for_reserve(
        std::int64_t cost_units,
        std::chrono::steady_clock::duration max_wait = std::chrono::steady_clock::duration::max());

    /// Record an OOM on `gpu_idx`: multiply that GPU's K by oom_backoff
    /// so future estimates over-book less aggressively. Idempotent-ish;
    /// multiple OOMs racing converge monotonically.
    void record_oom(int gpu_idx);

    [[nodiscard]] std::size_t num_gpus() const noexcept { return m_gpus.size(); }

    /// Read-only snapshots for benchmarking / diagnostics.
    [[nodiscard]] std::int64_t budget(int gpu_idx) const noexcept
    {
      return m_gpus[static_cast<std::size_t>(gpu_idx)].budget;
    }
    [[nodiscard]] std::int64_t remaining(int gpu_idx) const noexcept
    {
      return m_gpus[static_cast<std::size_t>(gpu_idx)].remaining.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double k_bytes_per_unit(int gpu_idx) const noexcept
    {
      return m_gpus[static_cast<std::size_t>(gpu_idx)].k_bytes.load(std::memory_order_relaxed);
    }
    [[nodiscard]] int active_count() const noexcept
    {
      return m_active.load(std::memory_order_relaxed);
    }
    [[nodiscard]] int max_concurrent() const noexcept { return m_max_concurrent; }

    /// Cumulative counters since construction. Useful for benchmarking
    /// without poking at scheduler internals; also snapshotted into
    /// `last_gpu_scheduler_stats()` on destruction.
    [[nodiscard]] std::int64_t total_admitted() const noexcept
    {
      return m_total_admitted.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::int64_t total_failed_no_room() const noexcept
    {
      return m_total_failed_no_room.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::int64_t total_failed_cap() const noexcept
    {
      return m_total_failed_cap.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::int64_t total_oom() const noexcept
    {
      return m_total_oom.load(std::memory_order_relaxed);
    }
    [[nodiscard]] int peak_active() const noexcept
    {
      return m_peak_active.load(std::memory_order_relaxed);
    }

    /// Record an empirical per-simplex byte count observed for an item
    /// that ran on `gpu_idx`. Raises `k_bytes` for that GPU to
    /// `max(current, k_obs)` without ever shrinking it -- AIMD-style
    /// safety: we trust observed usage that exceeds our estimate but
    /// do not trust observations that are lower than it (those are
    /// dominated by noise when multiple items share the device).
    ///
    /// The hybrid dispatcher calls this on the first admitted item per
    /// GPU based on `cudaMemGetInfo` snapshots taken on admit and
    /// release; after that the per-GPU `calibrated` flag is set and
    /// further calibration measurements are skipped (AIMD via
    /// `record_oom` still applies). Exposed publicly so tests can
    /// drive it deterministically without needing a live GPU.
    void note_observed_k(int gpu_idx, double k_obs) noexcept;

    [[nodiscard]] bool is_calibrated(int gpu_idx) const noexcept
    {
      return m_gpus[static_cast<std::size_t>(gpu_idx)].calibrated.load(std::memory_order_relaxed);
    }

  private:
    struct GpuState
    {
      std::int64_t budget = 0;
      std::atomic<std::int64_t> remaining{0};
      std::atomic<double> k_bytes{1.0};
      /// Set once the first admitted item on this GPU has been
      /// measured via `cudaMemGetInfo` snapshots. Further admissions
      /// skip the measurement.
      std::atomic<bool> calibrated{false};
    };

    void release_bytes(Reservation& r) noexcept;

    void update_peak_active(int now_active) noexcept;

    std::vector<GpuState> m_gpus;
    double m_oom_backoff;
    int m_max_concurrent;
    /// True on the live-device constructor path, false on the
    /// test-injected path: enables first-admit cudaMemGetInfo snapshot.
    bool m_calibrate_enabled = false;
    std::atomic<int> m_active{0};
    std::atomic<int> m_peak_active{0};
    std::atomic<std::int64_t> m_total_admitted{0};
    std::atomic<std::int64_t> m_total_failed_no_room{0};
    std::atomic<std::int64_t> m_total_failed_cap{0};
    std::atomic<std::int64_t> m_total_oom{0};

    mutable std::mutex m_wait_mutex;
    std::condition_variable m_wait_cv;
  };

  inline void GpuMemoryScheduler::Reservation::release() noexcept
  {
    if (m_sched) {
      m_sched->release_bytes(*this);
      m_sched = nullptr;
      m_gpu = -1;
      m_bytes = 0;
      m_cost_units = 0;
      m_free_before = 0;
      m_calibrate = false;
    }
  }

  inline GpuMemoryScheduler::GpuMemoryScheduler(Config cfg)
    : m_oom_backoff(cfg.oom_backoff), m_max_concurrent(cfg.max_concurrent),
      m_calibrate_enabled(true)
  {
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n < 0) n = 0;

    m_gpus = std::vector<GpuState>(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      cudaSetDevice(i);
      std::size_t free_mem = 0, total_mem = 0;
      cudaMemGetInfo(&free_mem, &total_mem);
      m_gpus[static_cast<std::size_t>(i)].budget =
        static_cast<std::int64_t>(static_cast<double>(free_mem) * cfg.budget_fraction);
      m_gpus[static_cast<std::size_t>(i)].remaining.store(
        m_gpus[static_cast<std::size_t>(i)].budget, std::memory_order_relaxed);
      m_gpus[static_cast<std::size_t>(i)].k_bytes.store(
        cfg.initial_k_bytes_per_unit, std::memory_order_relaxed);
    }
  }

  inline GpuMemoryScheduler::GpuMemoryScheduler(std::vector<std::int64_t> per_device_budgets, Config cfg)
    : m_gpus(per_device_budgets.size()), m_oom_backoff(cfg.oom_backoff),
      m_max_concurrent(cfg.max_concurrent), m_calibrate_enabled(false)
  {
    for (std::size_t i = 0; i < per_device_budgets.size(); ++i) {
      m_gpus[i].budget = per_device_budgets[i];
      m_gpus[i].remaining.store(per_device_budgets[i], std::memory_order_relaxed);
      m_gpus[i].k_bytes.store(cfg.initial_k_bytes_per_unit, std::memory_order_relaxed);
    }
  }

  inline GpuMemoryScheduler::Reservation
  GpuMemoryScheduler::try_reserve(std::int64_t cost_units)
  {
    if (cost_units <= 0) {
      return Reservation{};
    }

    // Tentatively claim a slot before walking GPUs. Always track
    // `m_active` so callers can observe live reservations for
    // diagnostics, even when no cap is set; if a cap is set, enforce it
    // here. Roll back below if no GPU has room.
    const int prev_active = m_active.fetch_add(1, std::memory_order_acquire);
    if (m_max_concurrent > 0 && prev_active >= m_max_concurrent) {
      m_active.fetch_sub(1, std::memory_order_release);
      m_total_failed_cap.fetch_add(1, std::memory_order_relaxed);
      return Reservation{};
    }

    for (std::size_t i = 0; i < m_gpus.size(); ++i) {
      auto& g = m_gpus[i];
      const double k = g.k_bytes.load(std::memory_order_relaxed);
      const std::int64_t est = static_cast<std::int64_t>(k * static_cast<double>(cost_units));

      // Item structurally too big for this GPU's total budget.
      if (est > g.budget) continue;

      const std::int64_t prev = g.remaining.fetch_sub(est, std::memory_order_acquire);
      if (prev >= est) {
        // The live constructor path calls cudaSetDevice here; the
        // test-injected budgets path still sets the device so callers
        // see consistent behavior across both constructors.
        cudaSetDevice(static_cast<int>(i));
        m_total_admitted.fetch_add(1, std::memory_order_relaxed);
        update_peak_active(prev_active + 1);

        // First-admit calibration snapshot: if we haven't yet calibrated
        // this GPU, capture its free memory now so release() can diff
        // against it and upgrade K to the observed bytes/unit. We only
        // take the shot if m_active is exactly 1 post-increment, i.e.
        // no other reservation is live; otherwise the measurement would
        // be contaminated by concurrent items.
        std::size_t free_before = 0;
        bool do_calibrate = false;
        if (m_calibrate_enabled && prev_active == 0 &&
            !g.calibrated.load(std::memory_order_acquire))
        {
          std::size_t total_mem = 0;
          if (cudaMemGetInfo(&free_before, &total_mem) == cudaSuccess) {
            do_calibrate = true;
          }
        }
        return Reservation{this, static_cast<int>(i), est,
                           cost_units, free_before, do_calibrate};
      }
      g.remaining.fetch_add(est, std::memory_order_release);
    }

    m_active.fetch_sub(1, std::memory_order_release);
    m_total_failed_no_room.fetch_add(1, std::memory_order_relaxed);
    return Reservation{};
  }

  inline void GpuMemoryScheduler::note_observed_k(int gpu_idx, double k_obs) noexcept
  {
    if (gpu_idx < 0 || !(k_obs > 0.0)) return;
    auto& k = m_gpus[static_cast<std::size_t>(gpu_idx)].k_bytes;
    double old_k = k.load(std::memory_order_relaxed);
    while (k_obs > old_k) {
      if (k.compare_exchange_weak(old_k, k_obs,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
        return;
      }
      // old_k updated by CAS; loop re-checks against fresh value.
    }
  }

  inline void GpuMemoryScheduler::update_peak_active(int now_active) noexcept
  {
    int peak = m_peak_active.load(std::memory_order_relaxed);
    while (now_active > peak &&
           !m_peak_active.compare_exchange_weak(peak, now_active,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
      // peak updated by CAS; loop to retry.
    }
  }

  inline GpuMemoryScheduler::Reservation
  GpuMemoryScheduler::wait_for_reserve(std::int64_t cost_units,
                                        std::chrono::steady_clock::duration max_wait)
  {
    if (cost_units <= 0) return Reservation{};

    // Structural-fit check: with current per-GPU K values, is there any
    // device whose budget could accommodate this item? If not, return
    // inactive immediately -- waiting would never succeed because K is
    // monotonically non-decreasing (AIMD only bumps up on OOM).
    auto fits_any = [this, cost_units]() {
      for (std::size_t i = 0; i < m_gpus.size(); ++i) {
        const double k = m_gpus[i].k_bytes.load(std::memory_order_relaxed);
        const auto est = static_cast<std::int64_t>(k * static_cast<double>(cost_units));
        if (est <= m_gpus[i].budget) return true;
      }
      return false;
    };

    if (!fits_any()) return Reservation{};

    using clock = std::chrono::steady_clock;
    const auto max_dur = clock::duration::max();
    const auto start = clock::now();
    const bool unbounded = (max_wait == max_dur);
    // Compute the deadline without overflow: if unbounded we never
    // check it, otherwise start + max_wait fits by construction.
    const auto deadline = unbounded ? clock::time_point::max() : start + max_wait;

    // Event-driven wait: release_bytes / record_oom synchronize via
    // m_wait_mutex (briefly) before notify_all, so a state change
    // visible to a subsequent try_reserve_locked must either happen
    // before our mutex acquire (and be observed by the re-try under
    // the lock) or after we enter wait_until (and wake us). No polling.
    while (true) {
      auto res = try_reserve(cost_units);
      if (res.active()) return res;

      if (!fits_any()) return Reservation{};

      std::unique_lock<std::mutex> lk(m_wait_mutex);
      // Re-try under the lock to close the race with a concurrent
      // release or OOM that completed between the try_reserve above
      // and our lock acquire. See release_bytes / record_oom for the
      // matching lock fence.
      res = try_reserve(cost_units);
      if (res.active()) return res;
      if (!fits_any()) return Reservation{};

      if (unbounded) {
        m_wait_cv.wait(lk);
      } else {
        if (clock::now() >= deadline) return Reservation{};
        if (m_wait_cv.wait_until(lk, deadline) == std::cv_status::timeout) {
          // One last try after timeout: a notify might have arrived
          // concurrently and flipped our state.
          lk.unlock();
          res = try_reserve(cost_units);
          if (res.active()) return res;
          return Reservation{};
        }
      }
    }
  }

  inline void GpuMemoryScheduler::record_oom(int gpu_idx)
  {
    auto& k = m_gpus[static_cast<std::size_t>(gpu_idx)].k_bytes;
    double old_k = k.load(std::memory_order_relaxed);
    while (true) {
      const double new_k = old_k * m_oom_backoff;
      if (k.compare_exchange_weak(old_k, new_k,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
        m_total_oom.fetch_add(1, std::memory_order_relaxed);
        // Brief mutex acquire then notify: waiters whose structural
        // fit just changed (K grew past budget / cost_units) must
        // re-check. The lock fence ensures our K update happens-before
        // any waiter that acquires m_wait_mutex after this notify.
        { std::lock_guard<std::mutex> g(m_wait_mutex); }
        m_wait_cv.notify_all();
        return;
      }
      // old_k updated by CAS; loop to retry with fresh value.
    }
  }

  inline GpuMemoryScheduler::~GpuMemoryScheduler()
  {
    LastSchedulerStats snap;
    snap.total_admitted = m_total_admitted.load(std::memory_order_relaxed);
    snap.total_failed_no_room = m_total_failed_no_room.load(std::memory_order_relaxed);
    snap.total_failed_cap = m_total_failed_cap.load(std::memory_order_relaxed);
    snap.total_oom = m_total_oom.load(std::memory_order_relaxed);
    snap.peak_active = m_peak_active.load(std::memory_order_relaxed);
    snap.num_gpus = m_gpus.size();

    std::lock_guard<std::mutex> g(last_gpu_scheduler_stats_mutex());
    last_gpu_scheduler_stats() = snap;
  }

  inline void GpuMemoryScheduler::release_bytes(Reservation& r) noexcept
  {
    if (r.m_gpu < 0 || r.m_bytes <= 0) return;
    auto& g = m_gpus[static_cast<std::size_t>(r.m_gpu)];

    // If this was the calibration item and it still has exclusive use
    // of the GPU at release time (m_active == 1, just us), snapshot
    // the free memory and back out K_obs = used_bytes / cost_units.
    // If another reservation came in since admit, the measurement
    // would be contaminated; skip it. Either way, the per-GPU
    // `calibrated` flag is set so we do not keep trying.
    if (r.m_calibrate) {
      const int now_active = m_active.load(std::memory_order_acquire);
      if (now_active == 1 && r.m_cost_units > 0) {
        std::size_t free_after = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_after, &total_mem) == cudaSuccess &&
            free_after < r.m_free_before)
        {
          const std::size_t used = r.m_free_before - free_after;
          const double k_obs =
            static_cast<double>(used) / static_cast<double>(r.m_cost_units);
          note_observed_k(r.m_gpu, k_obs);
        }
      }
      g.calibrated.store(true, std::memory_order_release);
    }

    g.remaining.fetch_add(r.m_bytes, std::memory_order_release);
    m_active.fetch_sub(1, std::memory_order_release);
    // Brief mutex acquire serializes the state change with waiters'
    // re-try-under-lock path in wait_for_reserve: either the waiter
    // has not yet acquired the mutex (its re-try_reserve sees the
    // freed bytes) or it is already inside wait_until (notify_all
    // wakes it). notify_all because different-sized waiters may be
    // queued and only some fit.
    { std::lock_guard<std::mutex> g(m_wait_mutex); }
    m_wait_cv.notify_all();
  }
}

#endif
