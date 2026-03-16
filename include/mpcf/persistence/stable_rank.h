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

#ifndef MASSPCF_STABLE_RANK_H
#define MASSPCF_STABLE_RANK_H

#include "../task.h"
#include "../functional/pcf.h"
#include "barcode.h"

namespace mpcf::ph
{
  /**
   * Converts a single barcode to a stable rank PCF. See Chachólski, W., & Riihimäki, H. (2020). Metrics and stabilization in one parameter persistence. SIAM Journal on Applied Algebra and Geometry, 4(1), 69-98.
   * @tparam T Data type of the bar birth/death values. The same type will be used for the PCF time and value types.
   * @param barcode The barcode to convert to a stable rank function.
   * @return The stable rank function corresponding to the supplied barcode.
   */
  template <typename T>
  Pcf<T, T> barcode_to_stable_rank(const Barcode<T>& barcode)
  {
    if (barcode.bars().empty())
    {
      return Pcf<T, T>();
    }

    std::vector<T> lifetimes;
    lifetimes.reserve(barcode.bars().size());

    std::transform(barcode.bars().begin(), barcode.bars().end(), std::back_inserter(lifetimes), [](const PersistencePair<T>& pp) -> T {
      if (pp.death == std::numeric_limits<T>::max())
      {
        return std::numeric_limits<T>::max();
      }

      return pp.death - pp.birth;
    });

    std::sort(lifetimes.begin(), lifetimes.end());

    using PcfT = Pcf<T, T>;
    using PcfPointT = typename PcfT::point_type;

    std::vector<PcfPointT> pcfPoints;

    T nAlive = static_cast<T>(lifetimes.size());

    pcfPoints.emplace_back(static_cast<T>(0.), nAlive);

    T lastLifetime = lifetimes.front();
    for (auto const & lifetime : lifetimes)
    {
      if (Barcode<T>::is_infinite(lastLifetime))
      {
        break;
      }

      if (lifetime != lastLifetime)
      {
        pcfPoints.emplace_back(lastLifetime, nAlive);
        lastLifetime = lifetime;
      }
      --nAlive;
    }

    if (!Barcode<T>::is_infinite(lastLifetime))
    {
      pcfPoints.emplace_back(lastLifetime, nAlive);
    }

    return Pcf<T, T>(std::move(pcfPoints));
  }

  template <typename T>
  class BarcodeToStableRankTask : public StoppableTask<void>
  {
  public:
    BarcodeToStableRankTask(const Tensor<Barcode<T>>& barcodes, Tensor<Pcf<T, T>>& ret)
        : m_barcodes(barcodes), m_ret(ret)
    {
    }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      tf::Taskflow flow;

      next_step(m_barcodes.size(), "Converting barcodes to stable rank functions", "barcode");

      m_ret = Tensor<Pcf<T, T>>(m_barcodes.shape());

      m_barcodes.walk([this, &flow](const std::vector<size_t>& index){

        auto task = flow.emplace([this, index]
        {
          m_ret(index) = barcode_to_stable_rank(m_barcodes(index));
          add_progress(1);
        });

      });

      return exec.cpu()->run(std::move(flow));
    }

    const Tensor<Barcode<T>>& m_barcodes;
    Tensor<Pcf<T, T>>& m_ret;
  };
}

#endif //MASSPCF_STABLE_RANK_H
