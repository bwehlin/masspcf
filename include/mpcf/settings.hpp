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

#ifndef MPCF_SETTINGS_HPP
#define MPCF_SETTINGS_HPP

#include <cstddef>

namespace mpcf
{
  struct Settings
  {
    bool forceCpu = false;
    size_t cudaThreshold = 500;
    bool deviceVerbose = false;

    unsigned int blockDimX = 1;
    unsigned int blockDimY = 32;

    /// Minimum block side length for the block scheduler.
    /// 0 = auto-detect from GPU hardware (SM count).
    size_t minBlockSide = 0;

    /// Hard cap on the number of concurrent GPU reservations the
    /// hybrid Ripser++ dispatcher will hand out across all GPUs.
    /// 0 = unlimited (the scheduler is bounded only by per-GPU memory).
    int gpuConcurrencyCap = 0;

    /// Fraction of free GPU memory (at scheduler construction) the
    /// hybrid Ripser++ dispatcher reserves as its scheduling budget.
    /// The remainder absorbs CUDA scratch, fragmentation, and other
    /// tenants. 0.6 is the scheduler's own default; valid range is
    /// 0 < f <= 1. Bumping this up trades safety headroom for more
    /// concurrency.
    double gpuBudgetFraction = 0.6;

    /// When the hybrid Ripser++ dispatcher cannot obtain a GPU slot
    /// (all slots busy or budget exhausted), should the item wait for
    /// a slot to free up (true) or immediately run on CPU (false)?
    /// Default false preserves the original hybrid behaviour. Queue
    /// policy is preferable when GPU is much faster than CPU per item
    /// (max_dim >= 2, where the apparent-pairs mechanism pays off);
    /// CPU fallback wins when the per-item speedup is small.
    /// OOM-triggered CPU fallback is unconditional regardless of this
    /// setting.
    bool hybridGpuQueueOnBusy = false;
  };

  inline Settings& settings()
  {
    static Settings instance;
    return instance;
  }
}

#endif
