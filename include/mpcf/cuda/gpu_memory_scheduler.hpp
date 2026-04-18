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
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mpcf
{
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
        : m_sched(o.m_sched), m_gpu(o.m_gpu), m_bytes(o.m_bytes)
      {
        o.m_sched = nullptr;
        o.m_gpu = -1;
        o.m_bytes = 0;
      }

      Reservation& operator=(Reservation&& o) noexcept
      {
        if (this != &o) {
          release();
          m_sched = o.m_sched;
          m_gpu = o.m_gpu;
          m_bytes = o.m_bytes;
          o.m_sched = nullptr;
          o.m_gpu = -1;
          o.m_bytes = 0;
        }
        return *this;
      }

      ~Reservation() { release(); }

      [[nodiscard]] bool active() const noexcept { return m_sched != nullptr; }
      [[nodiscard]] int gpu_index() const noexcept { return m_gpu; }
      [[nodiscard]] std::int64_t bytes() const noexcept { return m_bytes; }

    private:
      friend class GpuMemoryScheduler;
      Reservation(GpuMemoryScheduler* s, int gpu, std::int64_t bytes)
        : m_sched(s), m_gpu(gpu), m_bytes(bytes) {}

      void release() noexcept;

      GpuMemoryScheduler* m_sched = nullptr;
      int m_gpu = -1;
      std::int64_t m_bytes = 0;
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

    /// Probe GPUs in order 0..N-1 for a slot big enough for `cost_units`.
    /// On success: cudaSetDevice is called and a Reservation is returned.
    /// On failure (no GPU has enough room, or cost exceeds every GPU's
    /// total budget): an inactive Reservation is returned.
    Reservation try_reserve(std::int64_t cost_units);

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

  private:
    struct GpuState
    {
      std::int64_t budget = 0;
      std::atomic<std::int64_t> remaining{0};
      std::atomic<double> k_bytes{1.0};
    };

    void release_bytes(int gpu_idx, std::int64_t bytes) noexcept;

    std::vector<GpuState> m_gpus;
    double m_oom_backoff;
  };

  inline void GpuMemoryScheduler::Reservation::release() noexcept
  {
    if (m_sched) {
      m_sched->release_bytes(m_gpu, m_bytes);
      m_sched = nullptr;
      m_gpu = -1;
      m_bytes = 0;
    }
  }

  inline GpuMemoryScheduler::GpuMemoryScheduler(Config cfg)
    : m_oom_backoff(cfg.oom_backoff)
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
    : m_gpus(per_device_budgets.size()), m_oom_backoff(cfg.oom_backoff)
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
        return Reservation{this, static_cast<int>(i), est};
      }
      g.remaining.fetch_add(est, std::memory_order_release);
    }
    return Reservation{};
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
        return;
      }
      // old_k updated by CAS; loop to retry with fresh value.
    }
  }

  inline void GpuMemoryScheduler::release_bytes(int gpu_idx, std::int64_t bytes) noexcept
  {
    if (gpu_idx < 0 || bytes <= 0) return;
    m_gpus[static_cast<std::size_t>(gpu_idx)].remaining.fetch_add(bytes, std::memory_order_release);
  }
}

#endif
