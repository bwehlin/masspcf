// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MASSPCF_RIPSERPP_HPP
#define MASSPCF_RIPSERPP_HPP

#include "../../executor.hpp"
#include "../../tensor.hpp"
#include "../persistence_pair.hpp"

#include <cstddef>
#include <vector>

namespace mpcf::ph::ripserpp
{
  // Public entry point into the ported Ripser++. Computes Vietoris-Rips
  // persistence barcodes up to dimension maxDim for the given point cloud
  // on the GPU.
  //   points: shape (n, d) -- n points in R^d
  //   out:    resized to maxDim + 1; out[k] holds the k-th homology pairs.
  //   exec:   CPU executor for the parallel-for loops inside the port.
  // Throws mpcf::cuda_error (see mpcf/cuda/cuda_util.cuh) on CUDA failures;
  // inspect code() == cudaErrorMemoryAllocation to detect OOM.
  template <typename T>
  void compute_barcodes_pcloud(
      const PointCloud<T>& points,
      std::size_t maxDim,
      std::vector<std::vector<PersistencePair<T>>>& out,
      mpcf::Executor& exec);
}

#endif // MASSPCF_RIPSERPP_HPP
