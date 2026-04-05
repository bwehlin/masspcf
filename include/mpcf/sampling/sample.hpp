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

#ifndef MPCF_SAMPLING_SAMPLE_H
#define MPCF_SAMPLING_SAMPLE_H

#include "sampler.hpp"

namespace mpcf::sampling
{

  /// Sample k points around each vantage point, weighted by a distance distribution.
  ///
  /// Convenience wrapper that builds a DistanceWeightedSampler from the source
  /// cloud and immediately samples. If you need to sample from the same source
  /// cloud multiple times, construct a DistanceWeightedSampler directly to
  /// avoid rebuilding the KD-tree.
  template <typename T, typename Dist, typename EngineT>
  IndexedPointCloudCollection<T> sample(
      const PointCloud<T>& source,
      const PointCloud<T>& vantage,
      size_t k,
      const Dist& dist,
      bool replace,
      const RandomGenerator<EngineT>& gen,
      Executor& exec,
      T radius = std::numeric_limits<T>::infinity(),
      T min_correction = T(0))
  {
    DistanceWeightedSampler<T> sampler(source);
    return sampler.sample(vantage, k, dist, replace, gen, exec, radius, min_correction);
  }

}

#endif
