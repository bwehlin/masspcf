/*
* Copyright 2024 Bjorn Wehlin
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

#include "pcf.h"

#include <vector>
#include <random>

#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xrandom.hpp>

namespace mpcf
{

  template <typename Tt, typename Tv, typename F>
  void noisy_function(xt::xarray<Pcf<Tt, Tv>>& out, size_t nPoints, F func, Tv noise = 0.1)
  {
    auto fs = xt::flatten(out);

    auto byTimeAscending = mpcf::OrderByTimeAscending<Tt, Tv>();

    for (auto& f : fs)
    {
      // TODO: this is quite inefficient getting a new array every time
      auto ts = xt::random::rand<Tt>({ nPoints });
      auto vnoises = xt::random::randn<Tv>({ nPoints }, 0., noise);

      std::vector<mpcf::Point<Tt, Tv>> pts;
      pts.resize(nPoints);
      std::transform(ts.begin(), ts.end(), pts.begin(), [](Tt t) { return mpcf::Point<Tt, Tv>(t, 0.); });
      std::sort(pts.begin(), pts.end(), byTimeAscending);

      for (size_t i = 0; i < pts.size(); ++i)
      {
        pts[i].v = func(pts[i].t) + vnoises(i);
      }

      f = mpcf::Pcf<Tt, Tv>(std::move(pts));
    }
  }
}

#endif
