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

#ifndef MPCF_SAMPLING_SAMPLING_RESULT_H
#define MPCF_SAMPLING_SAMPLING_RESULT_H

#include "indexed_point_cloud.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mpcf::sampling
{

  /// Post-hoc diagnostics for a sampling operation.
  ///
  /// All vectors are indexed by vantage point (length M).
  struct SamplingDiagnostics
  {
    /// Per-vantage acceptance rate: accepted / total_attempts.
    std::vector<double> acceptance_rate;

    /// Per-vantage total number of sample_one calls across all escalation stages.
    std::vector<size_t> total_attempts;

    /// Per-vantage flag: true if adaptive escalation was triggered (i.e. bias was introduced).
    std::vector<bool> biased;

    /// True if every vantage point was sampled without escalation (fully unbiased).
    bool all_exact() const
    {
      return std::none_of(biased.begin(), biased.end(), [](bool b) { return b; });
    }

    /// Number of vantage points where adaptive escalation was triggered.
    size_t biased_vantage_count() const
    {
      return static_cast<size_t>(std::count(biased.begin(), biased.end(), true));
    }
  };

  /// Sampling output: the sampled point cloud collection plus diagnostics.
  template <typename T>
  struct SamplingResult
  {
    IndexedPointCloudCollection<T> collection;
    SamplingDiagnostics diagnostics;
  };

}

#endif
