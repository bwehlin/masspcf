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

#ifndef MPCF_RANDOM_H
#define MPCF_RANDOM_H

#include "functional/pcf.hpp"
#include "tensor.hpp"
#include "walk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <random>

namespace mpcf
{

  template <typename Tt, typename Tv, typename F>
  void noisy_function(Tensor<Pcf<Tt, Tv>>& out, size_t nPoints, F func,
                      Tv noise = 0.1,
                      const DefaultRandomGenerator& gen = default_generator(),
                      Executor& exec = default_executor())
  {
    using PcfT = Pcf<Tt, Tv>;
    using PointT = typename PcfT::point_type;

    mpcf::parallel_walk(out, gen, [nPoints, noise, &func, &out](const std::vector<size_t>& idx, auto& engine) {

      std::uniform_real_distribution<Tt> tDist(static_cast<Tt>(0.), static_cast<Tt>(1.));
      std::normal_distribution<Tv> vDist(static_cast<Tv>(0.), noise);

      std::vector<Tt> randomTs(nPoints);
      std::vector<Tv> randomNoises(nPoints);

      std::generate(randomTs.begin(), randomTs.end(), [&engine, &tDist]{ return tDist(engine); });
      std::generate(randomNoises.begin(), randomNoises.end(), [&engine, &vDist]{ return vDist(engine); });

      std::sort(randomTs.begin(), randomTs.end());
      randomTs.front() = 0.;

      std::vector<PointT> pts;
      pts.resize(randomTs.size());
      for (auto i = 0_uz; i < randomTs.size(); ++i)
      {
        pts[i].t = randomTs[i];
        pts[i].v = func(randomTs[i]) + randomNoises[i];
      }

      pts.back().v = 0.;

      out(idx) = PcfT(std::move(pts));
    }, exec);
  }

  /// Generate random PCFs following Algorithm 1 from [Wehlin 2024].
  ///
  /// Each PCF has a random number of breakpoints drawn from U[nMin, nMax],
  /// breakpoint times drawn from |N(0, alpha²)|, random values from N(0, 1),
  /// and final value 0.
  ///
  /// @param fixedAlpha  Controls the variance of the breakpoint time
  ///   distribution.  When > 0, all PCFs use this value.  When <= 0,
  ///   alpha is drawn from |N(0, 1)| independently per PCF.
  template <typename Tt, typename Tv>
  void random_pcf(Tensor<Pcf<Tt, Tv>>& out, size_t nMin, size_t nMax,
                  Tt fixedAlpha = Tt(0),
                  const DefaultRandomGenerator& gen = default_generator(),
                  Executor& exec = default_executor())
  {
    using PcfT = Pcf<Tt, Tv>;
    using PointT = typename PcfT::point_type;

    mpcf::parallel_walk(out, gen,
        [nMin, nMax, fixedAlpha, &out](const std::vector<size_t>& idx, auto& engine) {

      std::uniform_int_distribution<size_t> nDist(nMin, nMax);
      std::normal_distribution<Tt> normalDist;

      size_t n = nDist(engine);

      Tt alpha = fixedAlpha;
      if (alpha <= Tt(0))
      {
        alpha = std::abs(normalDist(engine));
        if (alpha < std::numeric_limits<Tt>::epsilon())
          alpha = std::numeric_limits<Tt>::epsilon();
      }

      // Draw and sort n-1 times: |N(0,1)| * alpha, all positive
      std::vector<Tt> rawTimes(n - 1);
      for (auto& t : rawTimes)
        t = std::abs(normalDist(engine));
      std::sort(rawTimes.begin(), rawTimes.end());

      // Draw n-1 values from N(0,1)
      std::vector<Tv> vals(n - 1);
      for (auto& v : vals)
        v = static_cast<Tv>(normalDist(engine));

      // Build points: (0, v₀), (α|t₁|, v₁), ..., (α|t_{n-1}|, 0)
      std::vector<PointT> pts(n);
      pts[0] = {Tt(0), vals[0]};
      for (size_t i = 1; i < n - 1; ++i)
      {
        pts[i].t = alpha * rawTimes[i - 1];
        pts[i].v = vals[i];
      }
      pts[n - 1] = {alpha * rawTimes[n - 2], Tv(0)};

      out(idx) = PcfT(std::move(pts));
    }, exec);
  }

}

#endif
