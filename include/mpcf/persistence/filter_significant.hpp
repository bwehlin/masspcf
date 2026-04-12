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

#ifndef MASSPCF_FILTER_SIGNIFICANT_H
#define MASSPCF_FILTER_SIGNIFICANT_H

#include "../task.hpp"
#include "barcode.hpp"

#include <cmath>
#include <string>

namespace mpcf::ph
{
  /**
   * Filters a persistence barcode to retain only statistically significant
   * bars, using the universal null-distribution hypothesis test from
   * Bobrowski & Skraba (2023).
   *
   * The test uses multiplicative persistence (death/birth ratio) and the
   * universal Left-Gumbel null-distribution to assign p-values to each
   * bar, with Bonferroni correction for multiple testing.
   *
   * Bars with birth <= 0 or infinite death are always retained (they
   * cannot be tested under the multiplicative persistence framework).
   *
   * Currently assumes the Vietoris-Rips complex (scale factor A = 1).
   *
   * @tparam T Data type of the bar birth/death values.
   * @param barcode The barcode to filter.
   * @param alpha Significance level for the hypothesis test (default 0.05).
   * @return A new barcode containing only the significant bars.
   *
   * @see Bobrowski, O. & Skraba, P. (2023). A universal null-distribution
   *      for topological data analysis. Scientific Reports, 13, 12274.
   */
  template <typename T>
  Barcode<T> filter_significant_bars(const Barcode<T>& barcode, T alpha = static_cast<T>(0.05))
  {
    if (barcode.bars().empty())
    {
      return Barcode<T>();
    }

    // Euler-Mascheroni constant
    constexpr T euler_mascheroni = static_cast<T>(0.5772156649015329);

    // Scale factor: A = 1 for Rips, A = 0.5 for Cech
    constexpr T A = static_cast<T>(1.0);

    // Classify bars into always-keep and testable
    std::vector<size_t> always_keep;
    std::vector<size_t> testable;
    std::vector<T> log_log_pi;

    for (size_t i = 0; i < barcode.bars().size(); ++i)
    {
      const auto& bar = barcode.bars()[i];

      if (Barcode<T>::is_infinite(bar.death) || bar.birth <= static_cast<T>(0))
      {
        always_keep.push_back(i);
        continue;
      }

      T pi = bar.death / bar.birth;
      if (pi <= static_cast<T>(1))
      {
        // death <= birth: degenerate bar, treat as noise (skip)
        continue;
      }

      T llp = std::log(std::log(pi));
      if (std::isfinite(llp))
      {
        testable.push_back(i);
        log_log_pi.push_back(llp);
      }
      else
      {
        // log(log(pi)) is not finite (pi very close to 1): skip
        continue;
      }
    }

    if (testable.empty())
    {
      // Nothing to test -- return only the always-keep bars
      std::vector<PersistencePair<T>> result;
      result.reserve(always_keep.size());
      for (size_t idx : always_keep)
        result.push_back(barcode.bars()[idx]);
      return Barcode<T>(std::move(result));
    }

    // Compute L_bar = mean of log(log(pi)) values
    T L_bar = static_cast<T>(0);
    for (T val : log_log_pi)
      L_bar += val;
    L_bar /= static_cast<T>(log_log_pi.size());

    // B = -lambda - A * L_bar
    T B = -euler_mascheroni - A * L_bar;

    // Bonferroni-corrected threshold
    T threshold = alpha / static_cast<T>(testable.size());

    // Test each bar
    std::vector<PersistencePair<T>> result;
    result.reserve(always_keep.size());

    for (size_t idx : always_keep)
      result.push_back(barcode.bars()[idx]);

    for (size_t j = 0; j < testable.size(); ++j)
    {
      T ell = A * log_log_pi[j] + B;
      T p_value = std::exp(-std::exp(ell));

      if (p_value < threshold)
      {
        result.push_back(barcode.bars()[testable[j]]);
      }
    }

    return Barcode<T>(std::move(result));
  }

  /**
   * Task that applies filter_significant_bars across a tensor of barcodes
   * in parallel.
   */
  template <typename T>
  class BarcodeFilterTask : public StoppableTask<void>
  {
  public:
    BarcodeFilterTask(const Tensor<Barcode<T>>& barcodes, Tensor<Barcode<T>>& ret,
                      T alpha, std::string progressLabel)
        : m_barcodes(barcodes), m_ret(ret), m_alpha(alpha),
          m_progressLabel(std::move(progressLabel))
    {
    }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      tf::Taskflow flow;

      next_step(m_barcodes.size(), m_progressLabel, "barcode");

      m_ret = Tensor<Barcode<T>>(m_barcodes.shape());

      mpcf::walk(m_barcodes, [this, &flow](const std::vector<size_t>& index) {
        flow.emplace([this, index] {
          m_ret(index) = filter_significant_bars(m_barcodes(index), m_alpha);
          add_progress(1);
        });
      });

      return exec.cpu()->run(std::move(flow));
    }

    const Tensor<Barcode<T>>& m_barcodes;
    Tensor<Barcode<T>>& m_ret;
    T m_alpha;
    std::string m_progressLabel;
  };

  template <typename T>
  auto make_filter_significant_task(const Tensor<Barcode<T>>& barcodes,
                                    Tensor<Barcode<T>>& out,
                                    T alpha = static_cast<T>(0.05))
  {
    return std::make_unique<BarcodeFilterTask<T>>(
        barcodes, out, alpha,
        "Filtering significant bars");
  }

} // namespace mpcf::ph

#endif // MASSPCF_FILTER_SIGNIFICANT_H
