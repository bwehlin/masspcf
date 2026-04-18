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

#ifndef MASSPCF_COMPUTE_PERSISTENCE_H
#define MASSPCF_COMPUTE_PERSISTENCE_H

#include "../tensor.hpp"
#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../task.hpp"

#include "barcode.hpp"
#include "persistence_pair.hpp"

#include "ripser/ripser.hpp"

#include <iostream>
#include <type_traits>

namespace mpcf::ph
{
  namespace detail
  {
    /// Run Ripser on any distance-matrix-like object (must have .size() and operator()(i,j)).
    /// Computes the enclosing radius threshold, builds the compressed lower-triangular
    /// matrix, runs Ripser, and writes barcodes into ret at the given index.
    template <typename DistMatT, typename T>
    void run_ripser(const DistMatT& distanceMatrix, size_t n, Tensor<Barcode<T>>& ret, size_t maxDim, const std::vector<size_t>& index, bool reducedHomology)
    {
      // A single point has no pairwise distances, so ripser's compressed
      // distance matrix would be empty and init_rows() would dereference
      // invalid memory.  Handle this trivially: one essential H0 bar
      // (unreduced) or nothing (reduced), and empty bars in higher dims.
      if (n <= 1)
      {
        for (auto i = 0_uz; i < maxDim + 1; ++i)
        {
          auto retIdx = index;
          retIdx.back() = i;

          std::vector<PersistencePair<T>> bars;
          if (i == 0 && !reducedHomology && n == 1)
          {
            bars.emplace_back(T{0}, std::numeric_limits<T>::infinity());
          }
          ret(retIdx) = std::move(bars);
        }
        return;
      }

      rips::value_t threshold = std::numeric_limits<rips::value_t>::infinity();
      for (auto i = 0_uz; i < n; ++i)
      {
        auto r = -std::numeric_limits<rips::value_t>::infinity();
        for (auto j = 0_uz; j < n; ++j)
        {
          r = std::max(r, static_cast<rips::value_t>(distanceMatrix(i, j)));
        }
        threshold = std::min(threshold, r);
      }

      rips::value_t ratio = static_cast<rips::value_t>(1);
      rips::coefficient_t modulus = 2;

      rips::compressed_lower_distance_matrix dist(distanceMatrix);
      rips::ripser<rips::compressed_lower_distance_matrix> ripser(std::move(dist), maxDim, threshold, ratio, modulus);
      ripser.compute_barcodes();

      for (auto i = 0_uz; i < maxDim + 1; ++i)
      {
        auto const & intervals = ripser.get_intervals(i);
        auto retIdx = index;
        retIdx.back() = i;

        std::vector<PersistencePair<T>> bars;
        bars.reserve(intervals.size() + 1);

        // Ripser computes reduced homology. When unreduced homology is
        // requested, insert the essential H0 class (born at 0, never dies).
        if (i == 0 && !reducedHomology)
        {
          bars.emplace_back(T{0}, std::numeric_limits<T>::infinity());
        }

        for (auto const& rpair : intervals)
        {
          bars.emplace_back(static_cast<T>(rpair.birth), static_cast<T>(rpair.death));
        }

        ret(retIdx) = std::move(bars);
      }
    }

    template <typename T>
    void compute_persistence_euclidean_single_impl(const Tensor<PointCloud<T>>& pclouds, Tensor<Barcode<T>>& ret, size_t maxDim, const std::vector<size_t>& index, bool reducedHomology = false)
    {
      if (index.back() != 0)
      {
        return;
      }

      auto pcIdx = std::vector<size_t>(index.begin(), std::prev(index.end()));
      auto const & points = pclouds(pcIdx);

      if (points.rank() == 0 || std::any_of(points.shape().begin(), points.shape().end(), [](size_t v){ return v == 0; }))
      {
        return;
      }

      if (points.rank() != 2)
      {
        throw std::runtime_error("Point cloud at index " + index_to_string(pcIdx) + " has unexpected shape " +
                                 shape_to_string(points.shape()) + " (should be (m, n))");
      }

      std::vector<std::vector<rips::value_t>> rpoints;
      rpoints.reserve(points.shape(0));

      for (auto i = 0_uz; i < points.shape(0); ++i)
      {
        rpoints.emplace_back();
        auto & curRPoint = rpoints.back();
        curRPoint.resize(points.shape(1));
        for (auto j = 0_uz; j < points.shape(1); ++j)
        {
          curRPoint[j] = points({i, j});
        }
      }

      rips::euclidean_distance_matrix distanceMatrix(std::move(rpoints));
      run_ripser(distanceMatrix, points.shape(0), ret, maxDim, index, reducedHomology);
    }

    template <typename T>
    void compute_persistence_distmat_single_impl(const Tensor<DistanceMatrix<T>>& dmats, Tensor<Barcode<T>>& ret, size_t maxDim, const std::vector<size_t>& index, bool reducedHomology = false)
    {
      if (index.back() != 0)
      {
        return;
      }

      auto dmIdx = std::vector<size_t>(index.begin(), std::prev(index.end()));
      auto const & dmat = dmats(dmIdx);

      if (dmat.size() == 0)
      {
        return;
      }

      run_ripser(dmat, dmat.size(), ret, maxDim, index, reducedHomology);
    }
  }

  /// Parallel Ripser task, templated on the input element type (PointCloud<T> or DistanceMatrix<T>).
  template <typename ElemT, typename T>
  class RipserTaskImpl : public StoppableTask<void>
  {
  public:
    RipserTaskImpl(const Tensor<ElemT>& input, Tensor<Barcode<T>>& ret, size_t maxDim = 1, bool reducedHomology = false)
      : m_input(input), m_ret(ret), m_maxDim(maxDim), m_reducedHomology(reducedHomology)
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto shape = m_input.shape();
      shape.emplace_back(m_maxDim + 1);
      m_ret = Tensor<Barcode<T>>(shape);

      next_step(m_input.size(), "Computing persistence", "pointcloud");

      return mpcf::parallel_walk_async(m_input, [this](const std::vector<size_t>& index) {
        if (stop_requested())
          return;

        thread_local std::vector<size_t> retIdx;
        retIdx.resize(index.size() + 1);
        std::copy(index.begin(), index.end(), retIdx.begin());
        retIdx.back() = 0;

        if constexpr (std::is_same_v<ElemT, PointCloud<T>>)
          detail::compute_persistence_euclidean_single_impl(m_input, m_ret, m_maxDim, retIdx, m_reducedHomology);
        else
          detail::compute_persistence_distmat_single_impl(m_input, m_ret, m_maxDim, retIdx, m_reducedHomology);
        add_progress(1);
      }, exec);
    }

    const Tensor<ElemT>& m_input;
    Tensor<Barcode<T>>& m_ret;
    size_t m_maxDim;
    bool m_reducedHomology;

  };

  template <typename T>
  using RipserTask = RipserTaskImpl<PointCloud<T>, T>;

  template <typename T>
  using RipserDistMatTask = RipserTaskImpl<DistanceMatrix<T>, T>;

#ifdef BUILD_WITH_CUDA
}

#include "ripserpp/ripserpp.hpp"
#include "../walk.hpp"

namespace mpcf::ph
{
  /// GPU Ripser++ task. Iterates items serially and feeds each point cloud
  /// into the Ripser++ facade. The ported Ripser++ no longer has global
  /// state but still issues thrust calls on the default CUDA stream, so
  /// multiple GPU jobs would serialize on the device. Phase 3 of the
  /// integration plan adds per-instance streams and a hybrid dispatcher.
  template <typename T>
  class RipserPlusPlusTask : public StoppableTask<void>
  {
  public:
    RipserPlusPlusTask(const Tensor<PointCloud<T>>& input, Tensor<Barcode<T>>& ret, size_t maxDim = 1, bool reducedHomology = false)
      : m_input(input), m_ret(ret), m_maxDim(maxDim), m_reducedHomology(reducedHomology)
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto shape = m_input.shape();
      shape.emplace_back(m_maxDim + 1);
      m_ret = Tensor<Barcode<T>>(shape);
      m_exec = &exec;

      next_step(m_input.size(), "Computing persistence (GPU)", "pointcloud");

      tf::Taskflow flow;
      flow.emplace([this]() {
        walk(m_input, [this](const std::vector<size_t>& index) {
          if (stop_requested()) return;
          process_item(index);
          add_progress(1);
        });
      });
      return exec.cpu()->run(std::move(flow));
    }

    void process_item(const std::vector<size_t>& index)
    {
      const auto& points = m_input(index);

      if (points.rank() == 0 ||
          std::any_of(points.shape().begin(), points.shape().end(), [](size_t v){ return v == 0; }))
      {
        return;
      }

      if (points.rank() != 2)
      {
        throw std::runtime_error("Point cloud at index " + index_to_string(index) + " has unexpected shape " +
                                 shape_to_string(points.shape()) + " (should be (m, n))");
      }

      std::vector<std::vector<PersistencePair<T>>> bars;
      ripserpp::compute_barcodes_pcloud<T>(points, m_maxDim, bars, *m_exec);

      for (size_t k = 0; k < bars.size(); ++k)
      {
        auto retIdx = index;
        retIdx.emplace_back(k);

        auto& kBars = bars[k];
        if (k == 0 && !m_reducedHomology)
        {
          kBars.insert(kBars.begin(), PersistencePair<T>(T{0}, std::numeric_limits<T>::infinity()));
        }
        m_ret(retIdx) = std::move(kBars);
      }
    }

    const Tensor<PointCloud<T>>& m_input;
    Tensor<Barcode<T>>& m_ret;
    size_t m_maxDim;
    bool m_reducedHomology;
    Executor* m_exec = nullptr;
  };
#endif // BUILD_WITH_CUDA
}

#endif //MASSPCF_COMPUTE_PERSISTENCE_H