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

#include "../tensor.h"
#include "../executor.h"
#include "../task.h"

#include "barcode.h"
#include "persistence_pair.h"

#include "ripser/ripser.h"

#include <iostream>

namespace mpcf::ph
{
  namespace detail
  {
    template <typename T>
    void compute_persistence_euclidean_single_impl(const Tensor<PointCloud<T>>& pclouds, Tensor<Barcode<T>>& ret, size_t maxDim, const std::vector<size_t>& index)
    {
      if (index.back() != 0)
      {
        // We do the computation on the index that corresponds to H_0. "H_0" writes into all H_k.
        return;
      }

      auto pcIdx = std::vector<size_t>(index.begin(), std::prev(index.end()));
      auto const & points = pclouds(pcIdx);

      if (points.rank() == 0 || std::any_of(points.shape().begin(), points.shape().end(), [](size_t v){ return v == 0; }))
      {
        // No points or all degenerate points => trivial homology
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

      rips::value_t threshold = std::numeric_limits<rips::value_t>::infinity();
      for (auto i = 0_uz; i < points.shape(0); ++i)
      {
        auto r = -std::numeric_limits<rips::value_t>::infinity();
        for (auto j = 0_uz; j < points.shape(0); ++j)
        {
          r = std::max(r, distanceMatrix(i, j));
        }
        threshold = std::min(threshold, r);
      }

      rips::value_t ratio = static_cast<rips::value_t>(1);
      rips::coefficient_t modulus = 2;

      rips::compressed_lower_distance_matrix dist(std::move(distanceMatrix));
      rips::ripser<rips::compressed_lower_distance_matrix> ripser(std::move(dist), maxDim, threshold, ratio, modulus);
      ripser.compute_barcodes();

      for (auto i = 0_uz; i < maxDim + 1; ++i)
      {
        auto const & intervals = ripser.get_intervals(i);
        auto retIdx = index;
        retIdx.back() = i;

        if constexpr (std::is_same_v<T, rips::value_t>)
        {
          ret(retIdx) = intervals;
        }
        else
        {
          // TODO: template on bar type in Ripser so that we don't have to do this conversion
          std::vector<PersistencePair<T>> conv;
          conv.resize(intervals.size());
          std::transform(intervals.begin(), intervals.end(), conv.begin(),
            [](const PersistencePair<rips::value_t>& rpair) {
              return PersistencePair<T>(static_cast<T>(rpair.birth), static_cast<T>(rpair.death));
            });
          ret(retIdx) = conv;
        }
      }
    }
  }

  template <typename T>
  class RipserTask : public StoppableTask<void>
  {
  public:
    RipserTask(const Tensor<PointCloud<T>>& pclouds, Tensor<Barcode<T>>& ret, size_t maxDim = 1)
      : m_pclouds(pclouds), m_ret(ret), m_maxDim(maxDim)
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      tf::Taskflow flow;

      auto shape = m_pclouds.shape();
      shape.emplace_back(m_maxDim + 1);
      m_ret = Tensor<Barcode<T>>(shape);

      next_step(m_pclouds.size(), "Computing persistence", "pointcloud");

      m_ret.walk([this, &flow](const std::vector<size_t>& index) {
        if (index.back() != 0)
        {
          // We do the computation on the index that corresponds to H_0. "H_0" writes into all H_k.
          return;
        }

        flow.emplace([this, index] {
          if (stop_requested())
            return;
          detail::compute_persistence_euclidean_single_impl(m_pclouds, m_ret, m_maxDim, index);
          add_progress(1);
        });
      });

      return exec.cpu()->run(std::move(flow));
    }

    const Tensor<PointCloud<T>>& m_pclouds;
    Tensor<Barcode<T>>& m_ret;
    size_t m_maxDim;

  };



}

#endif //MASSPCF_COMPUTE_PERSISTENCE_H