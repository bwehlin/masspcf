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

#include <vector>
#include <random>

#include "tensor.hpp"

namespace mpcf
{

  template <typename Tt, typename Tv, typename F>
  void noisy_function(Tensor<Pcf<Tt, Tv>>& out, size_t nPoints, F func, Tv noise = 0.1)
  {
    using PcfT = Pcf<Tt, Tv>;
    using PointT = typename PcfT::point_type;

    // TODO: unified random generator with seeding and thread-consistency

    std::mt19937_64 gen;
    std::uniform_real_distribution<Tt> tDist(static_cast<Tt>(0.), static_cast<Tt>(1.));
    std::normal_distribution<Tv> vDist(static_cast<Tv>(0.), noise);

    std::vector<Tt> randomTs(nPoints, static_cast<Tt>(0.));
    std::vector<Tv> randomNoises(nPoints, static_cast<Tt>(0.));

    out.apply([&gen, &tDist, &vDist, &randomTs, &randomNoises, &func](PcfT& f) {

      std::generate(randomTs.begin(), randomTs.end(), [&gen, &tDist]{ return tDist(gen); });
      std::generate(randomNoises.begin(), randomNoises.end(), [&gen, &vDist]{ return vDist(gen); });

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

      f = PcfT(std::move(pts));
    });
  }

}

#endif
