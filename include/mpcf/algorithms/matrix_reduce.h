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

#ifndef MPCF_ALGORITHMS_MATRIX_REDUCE_H
#define MPCF_ALGORITHMS_MATRIX_REDUCE_H

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xview.hpp>

#include <iostream>

#include "../pcf.h"
#include "../executor.h"

namespace mpcf
{
  template <typename XArrayT, typename XExpressionT>
  XArrayT parallel_matrix_reduce(const XExpressionT& in, size_t dim, Executor& exec = default_executor())
  {
    using pcf_type = typename XArrayT::value_type;
    using pcf_value_type = typename pcf_type::value_type;

    auto inBegin = xt::axis_begin(in, dim);
    auto inEnd = xt::axis_end(in, dim);

    XArrayT ret = xt::zeros_like(*inBegin);

    auto retFlat = xt::flatten(ret);
    auto flatShape = retFlat.shape(0);

    for (auto it = inBegin; it != inEnd; ++it)
    {
      auto flat = xt::flatten(*it);
      for (auto i = 0; i < flatShape; ++i)
      {
        retFlat[{i}] += flat.at(i);
      }
    }

    for (auto i = 0; i < retFlat.shape()[{0}]; ++i)
    {
      retFlat[{i}] /= static_cast<pcf_value_type>(std::distance(xt::axis_begin(in, dim), xt::axis_end(in, dim)));
    }

    return ret;
  }
}

#endif
