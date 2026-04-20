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

#include "../../distance_matrix.hpp"
#include "../../executor.hpp"
#include "../../tensor.hpp"
#include "../persistence_pair.hpp"

#include <cstddef>
#include <vector>

namespace mpcf::ph::ripserpp
{
  /// Diagnostic information collected during a single Ripser++ invocation.
  /// Problem-agnostic with respect to the caller: callers decide what to
  /// do with these signals (scheduler K bumps, statistics, retries).
  struct Diagnostics
  {
    /// True iff the ported Ripser++ fell back to CPU-only for part of
    /// the high-dimensional computation because the GPU memory it saw
    /// at construction was insufficient for the requested max dimension.
    /// The output barcodes are still correct; the wall time includes
    /// both the attempted GPU work and the CPU tail.
    bool upstream_cpu_fallback = false;

    /// The actual max dimension ripser++ ran on the GPU. Equals the
    /// caller's maxDim in the no-fallback path. Less than maxDim when
    /// the embedded memory planner lowered it.
    std::size_t gpu_max_dim = 0;
  };

  // Public entry point into the ported Ripser++. Computes Vietoris-Rips
  // persistence barcodes up to dimension maxDim for the given point cloud
  // on the GPU.
  //   points:          shape (n, d) -- n points in R^d
  //   out:             resized to maxDim + 1; out[k] holds the k-th
  //                    homology pairs.
  //   exec:            CPU executor for the parallel-for loops inside
  //                    the port (only used when `parallel_inner_loops`).
  //   diag:            optional -- populated with per-invocation diagnostics.
  //   parallel_inner_loops:
  //                    if true (default), ripser++'s internal
  //                    parallel-for loops run across `exec.cpu()`; if
  //                    false, those loops run serially on the calling
  //                    thread. The hybrid dispatcher passes false for
  //                    batch sizes > 1 so parallelism comes from
  //                    running many ripser instances concurrently
  //                    instead of subdividing each one.
  // Throws mpcf::cuda_error (see mpcf/cuda/cuda_util.cuh) on CUDA failures;
  // inspect code() == cudaErrorMemoryAllocation to detect OOM.
  template <typename T>
  void compute_barcodes_pcloud(
      const PointCloud<T>& points,
      std::size_t maxDim,
      std::vector<std::vector<PersistencePair<T>>>& out,
      mpcf::Executor& exec,
      Diagnostics* diag = nullptr,
      bool parallel_inner_loops = true);

  // Dense distance-matrix entry point. Mirrors `compute_barcodes_pcloud`
  // but takes precomputed pairwise distances; useful when the user has
  // a non-Euclidean metric or already has a DistanceMatrix at hand.
  //   dmat:            size n x n lower-triangular compressed distance matrix.
  // Sparse distance-matrix input is not supported.
  template <typename T>
  void compute_barcodes_distmat(
      const DistanceMatrix<T>& dmat,
      std::size_t maxDim,
      std::vector<std::vector<PersistencePair<T>>>& out,
      mpcf::Executor& exec,
      Diagnostics* diag = nullptr,
      bool parallel_inner_loops = true);
}

#endif // MASSPCF_RIPSERPP_HPP
