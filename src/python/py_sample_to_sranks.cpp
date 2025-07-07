/*
* Copyright 2024-2025 Bjorn Wehlin
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

#include "py_sample_to_sranks.h"

#include <pybind11/numpy.h>

#include <vector>
#include <iostream>

#include <mpcf/pcf.h>
#include <mpcf/executor.h>
#include <taskflow/algorithm/for_each.hpp>

#include <unordered_map>
#include <mpcf/persistence/ripser.h>

#include "pyarray.h"

using FloatT = double;
using IntT = long long int;

namespace py = pybind11;

namespace mpcf_py
{
  using Tt = float;
  using Tv = float;

  FloatT compute_distance(py::array_t<FloatT>& data, py::array_t<IntT>& sample, int dataId, int sampleId, int from, int to)
  {
    auto fromId = sample.template unchecked<3>()(dataId, sampleId, from);
    auto toId = sample.template unchecked<3>()(dataId, sampleId, to);

    auto nDataDim = data.shape(1);
    auto dataArr = data.template unchecked<2>();

    FloatT dist = 0.;
    for (auto iDim = 0; iDim < nDataDim; ++iDim)
    {
      auto diff = dataArr(fromId, iDim) - dataArr(toId, iDim);
      dist += diff * diff;
    }

    return std::sqrt(dist);
  }

  mpcf::Pcf<Tt, Tv> to_srank(const std::vector<mpcf::PersistencePair>& bars)
  {
    std::vector<Tt> lifetimes;
    lifetimes.reserve(bars.size());
    int nInfinite = 0;
    for (auto const & bar : bars)
    {
      if (bar.isDeathFinite())
      {
        lifetimes.emplace_back(bar.death - bar.birth);
      }
      else
      {
        ++nInfinite;
      }
    }
    std::sort(lifetimes.begin(), lifetimes.end());

    std::vector<mpcf::Point<Tt, Tv>> points;
    points.emplace_back(0., bars.size());
    if (lifetimes.empty())
    {
      return mpcf::Pcf<Tt, Tv>(std::move(points)); // All infinite, or zero PCF
    }

    auto lastLifetime = lifetimes[0];
    int surviving = bars.size();
    for (auto const & lifetime : lifetimes)
    {
      if (lastLifetime != lifetime)
      {
        points.emplace_back(lastLifetime, static_cast<Tv>(surviving));
        lastLifetime = lifetime;
      }
      --surviving;
    }
    points.emplace_back(lastLifetime, static_cast<Tv>(surviving));

    return mpcf::Pcf<Tt, Tv>(std::move(points));
  }

  void extract_srank_id(py::array_t<FloatT>& data, py::array_t<IntT>& sample, int dataId, NdArray<Tt, Tv>& arr, int maxDim, tf::Executor& /*exec*/)
  {
    size_t nSamples = sample.shape(1);
    size_t nSampleSize = sample.shape(2);

    for (size_t sampleId = 0; sampleId < nSamples; ++sampleId)
    {
      std::vector<FloatT> distances;

      FloatT threshold = 0.;
      distances.resize((nSampleSize - 1) * nSampleSize / 2);

      size_t idx = 0;
      for (size_t i = 1ul; i < nSampleSize; ++i)
      {
        for (size_t j = 0ul; j < i; ++j)
        {
          auto dist = compute_distance(data, sample, dataId, sampleId, i, j);
          distances[idx++] = dist;
          threshold = std::max(threshold, dist);
        }
      }

      rp::compressed_lower_distance_matrix distanceMatrix(std::move(distances));
      rp::ripser<rp::compressed_lower_distance_matrix> ripser(std::move(distanceMatrix), maxDim, threshold, 1, 2);
      ripser.compute_barcodes();

      for (auto iDim = 0; iDim <= maxDim; ++iDim)
      {
        auto & pcf = arr.data()(iDim, dataId, sampleId);
        pcf = to_srank(ripser.get_intervals(iDim));
      }

    }
  }



  NdArray<Tt, Tv> sample_to_sranks(py::array_t<FloatT> data, py::array_t<IntT> sample, int maxDim)
  {
    auto & exec = mpcf::default_executor();

    size_t nDataPts = data.shape(0);
    size_t nSamples = sample.shape(1);

    NdArray<Tt, Tv> arr = NdArray<Tt, Tv>::make_zeros(Shape({static_cast<size_t>(maxDim + 1), nDataPts, nSamples}));

    tf::Taskflow flow;
    flow.for_each_index(ssize_t(0), sample.shape(0), ssize_t(1), [&data, &sample, &arr, maxDim, &exec](int id){ extract_srank_id(data, sample, id, arr, maxDim, *exec.cpu()); });

    exec.cpu()->run(std::move(flow));
    exec.cpu()->wait_for_all();

    return arr;
  }

  void register_sample_to_sranks(py::module_& m)
  {
    m.def("sample_to_sranks_64", &mpcf_py::sample_to_sranks);
  }

}

