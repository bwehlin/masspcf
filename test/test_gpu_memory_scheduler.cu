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

#include <gtest/gtest.h>

#include <mpcf/cuda/gpu_memory_scheduler.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace
{
  using mpcf::GpuMemoryScheduler;

  GpuMemoryScheduler::Config default_cfg()
  {
    GpuMemoryScheduler::Config cfg;
    cfg.initial_k_bytes_per_unit = 1.0;
    cfg.oom_backoff = 1.5;
    return cfg;
  }

  TEST(GpuMemoryScheduler, NoDevicesGivesInactiveReservations)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{}, default_cfg());
    EXPECT_EQ(sched.num_gpus(), 0u);

    auto r = sched.try_reserve(100);
    EXPECT_FALSE(r.active());
  }

  TEST(GpuMemoryScheduler, ZeroOrNegativeCostYieldsInactive)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{1000}, default_cfg());

    auto r0 = sched.try_reserve(0);
    EXPECT_FALSE(r0.active());

    auto rneg = sched.try_reserve(-5);
    EXPECT_FALSE(rneg.active());

    EXPECT_EQ(sched.remaining(0), 1000);
  }

  TEST(GpuMemoryScheduler, ReserveDeductsFromRemaining)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{1000}, default_cfg());

    auto r = sched.try_reserve(100);
    ASSERT_TRUE(r.active());
    EXPECT_EQ(r.gpu_index(), 0);
    EXPECT_EQ(r.bytes(), 100);
    EXPECT_EQ(sched.remaining(0), 900);
  }

  TEST(GpuMemoryScheduler, DestructorReleasesBudget)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{1000}, default_cfg());

    {
      auto r = sched.try_reserve(250);
      ASSERT_TRUE(r.active());
      EXPECT_EQ(sched.remaining(0), 750);
    }
    EXPECT_EQ(sched.remaining(0), 1000);
  }

  TEST(GpuMemoryScheduler, OversizedItemReturnsInactive)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{100, 200}, default_cfg());

    auto r = sched.try_reserve(1000);
    EXPECT_FALSE(r.active());
    EXPECT_EQ(sched.remaining(0), 100);
    EXPECT_EQ(sched.remaining(1), 200);
  }

  TEST(GpuMemoryScheduler, FirstFitAcrossMultipleGpus)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{100, 1000}, default_cfg());

    auto r = sched.try_reserve(500);
    ASSERT_TRUE(r.active());
    EXPECT_EQ(r.gpu_index(), 1);
    EXPECT_EQ(sched.remaining(0), 100);
    EXPECT_EQ(sched.remaining(1), 500);
  }

  TEST(GpuMemoryScheduler, RunningOutOfEverythingGivesInactive)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{100, 100}, default_cfg());

    auto r1 = sched.try_reserve(80);
    ASSERT_TRUE(r1.active());
    EXPECT_EQ(r1.gpu_index(), 0);

    auto r2 = sched.try_reserve(80);
    ASSERT_TRUE(r2.active());
    EXPECT_EQ(r2.gpu_index(), 1);

    auto r3 = sched.try_reserve(80);
    EXPECT_FALSE(r3.active());

    EXPECT_EQ(sched.remaining(0), 20);
    EXPECT_EQ(sched.remaining(1), 20);
  }

  TEST(GpuMemoryScheduler, OomBumpsKForThatGpuOnly)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{1000, 1000}, default_cfg());
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(0), 1.0);
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(1), 1.0);

    sched.record_oom(0);
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(0), 1.5);
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(1), 1.0);

    sched.record_oom(0);
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(0), 2.25);
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(1), 1.0);
  }

  TEST(GpuMemoryScheduler, RaisedKReducesAvailableSlotsOnThatGpu)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{100}, default_cfg());

    {
      auto r = sched.try_reserve(50);
      ASSERT_TRUE(r.active());
      EXPECT_EQ(r.bytes(), 50);
    }
    EXPECT_EQ(sched.remaining(0), 100);

    sched.record_oom(0);
    EXPECT_DOUBLE_EQ(sched.k_bytes_per_unit(0), 1.5);

    auto r1 = sched.try_reserve(50);
    ASSERT_TRUE(r1.active());
    EXPECT_EQ(r1.bytes(), 75);
    EXPECT_EQ(sched.remaining(0), 25);

    auto r2 = sched.try_reserve(50);
    EXPECT_FALSE(r2.active());
  }

  TEST(GpuMemoryScheduler, MaxConcurrentCapBlocksFurtherReservations)
  {
    auto cfg = default_cfg();
    cfg.max_concurrent = 2;
    GpuMemoryScheduler sched(std::vector<std::int64_t>{1000}, cfg);

    auto r1 = sched.try_reserve(10);
    ASSERT_TRUE(r1.active());
    auto r2 = sched.try_reserve(10);
    ASSERT_TRUE(r2.active());
    EXPECT_EQ(sched.active_count(), 2);

    auto r3 = sched.try_reserve(10);
    EXPECT_FALSE(r3.active());
    EXPECT_EQ(sched.active_count(), 2);
    EXPECT_EQ(sched.remaining(0), 980);

    // The cap-rejected reservation must be visible in the cap counter,
    // not the no-room counter.
    EXPECT_EQ(sched.total_failed_cap(), 1);
    EXPECT_EQ(sched.total_failed_no_room(), 0);
    EXPECT_EQ(sched.total_admitted(), 2);
  }

  TEST(GpuMemoryScheduler, MaxConcurrentCapReleasesOnDestruction)
  {
    auto cfg = default_cfg();
    cfg.max_concurrent = 1;
    GpuMemoryScheduler sched(std::vector<std::int64_t>{1000}, cfg);

    {
      auto r = sched.try_reserve(10);
      ASSERT_TRUE(r.active());
      EXPECT_EQ(sched.active_count(), 1);

      auto r2 = sched.try_reserve(10);
      EXPECT_FALSE(r2.active());
    }
    EXPECT_EQ(sched.active_count(), 0);

    auto r3 = sched.try_reserve(10);
    EXPECT_TRUE(r3.active());
  }

  TEST(GpuMemoryScheduler, CumulativeCountersTrackOutcomes)
  {
    GpuMemoryScheduler sched(std::vector<std::int64_t>{200}, default_cfg());

    auto r1 = sched.try_reserve(100);
    ASSERT_TRUE(r1.active());
    auto r2 = sched.try_reserve(100);
    ASSERT_TRUE(r2.active());
    auto r3 = sched.try_reserve(100);  // bin drained -> failed
    EXPECT_FALSE(r3.active());

    EXPECT_EQ(sched.total_admitted(), 2);
    EXPECT_EQ(sched.total_failed_no_room(), 1);
    EXPECT_EQ(sched.total_oom(), 0);
    EXPECT_GE(sched.peak_active(), 2);

    sched.record_oom(0);
    sched.record_oom(0);
    EXPECT_EQ(sched.total_oom(), 2);

    // cost_units <= 0 should NOT count as a "no room" failure.
    auto rzero = sched.try_reserve(0);
    EXPECT_FALSE(rzero.active());
    EXPECT_EQ(sched.total_failed_no_room(), 1);
  }

  TEST(GpuMemoryScheduler, DestructorSnapshotsCountersToGlobal)
  {
    mpcf::last_gpu_scheduler_stats() = mpcf::LastSchedulerStats{};

    {
      GpuMemoryScheduler sched(std::vector<std::int64_t>{1000}, default_cfg());
      auto r1 = sched.try_reserve(300);
      ASSERT_TRUE(r1.active());
      auto r2 = sched.try_reserve(300);
      ASSERT_TRUE(r2.active());
      auto r3 = sched.try_reserve(900);  // doesn't fit -> failed
      EXPECT_FALSE(r3.active());
      sched.record_oom(0);
    }

    const auto& snap = mpcf::last_gpu_scheduler_stats();
    EXPECT_EQ(snap.total_admitted, 2);
    EXPECT_EQ(snap.total_failed_no_room, 1);
    EXPECT_EQ(snap.total_oom, 1);
    EXPECT_GE(snap.peak_active, 2);
    EXPECT_EQ(snap.num_gpus, 1u);
  }

  TEST(GpuMemoryScheduler, ConcurrentReservationsHeldDoNotOverbook)
  {
    constexpr std::int64_t budget = 1000;
    constexpr std::int64_t cost = 100;
    constexpr int num_threads = 64;

    GpuMemoryScheduler sched(std::vector<std::int64_t>{budget}, default_cfg());

    std::atomic<int> success_count{0};
    std::atomic<std::int64_t> peak_held{0};
    std::atomic<std::int64_t> currently_held{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        while (!go.load(std::memory_order_acquire)) {}
        auto r = sched.try_reserve(cost);
        if (r.active()) {
          success_count.fetch_add(1, std::memory_order_relaxed);
          auto held = currently_held.fetch_add(cost, std::memory_order_relaxed) + cost;
          auto prev_peak = peak_held.load(std::memory_order_relaxed);
          while (held > prev_peak &&
                 !peak_held.compare_exchange_weak(prev_peak, held,
                                                  std::memory_order_relaxed)) {}
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          currently_held.fetch_sub(cost, std::memory_order_relaxed);
        }
      });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    EXPECT_LE(peak_held.load(), budget);
    EXPECT_EQ(success_count.load(), budget / cost);
    EXPECT_EQ(sched.remaining(0), budget);
  }
}
