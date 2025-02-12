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

#ifndef MPCF_ALGORITHMS_MATRIX_REDUCE_H
#define MPCF_ALGORITHMS_MATRIX_REDUCE_H

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xview.hpp>

#include <taskflow/algorithm/for_each.hpp>

#include "../pcf.h"
#include "../executor.h"
#include "reduce.h"

namespace mpcf
{
  template <typename PcfT>
  struct TimeOpMaxTime
  {
    using time_type = typename PcfT::time_type;
    time_type get_time_of_interest(const PcfT& f) const
    {
      return f.points().back().t;
    }

    time_type operator()(time_type a, time_type b) const
    {
      return std::max(a, b);
    }

  };


  template <typename XExpressionT, typename Op>
  xt::xarray<typename XExpressionT::value_type::time_type> matrix_time_reduce(const XExpressionT& in, size_t dim, Op op /*, Executor& exec = default_executor() */)
  {
    using pcf_type = typename XExpressionT::value_type;
    using pcf_time_type = typename pcf_type::time_type;

    auto inBegin = xt::axis_begin(in, dim);
    auto inEnd = xt::axis_end(in, dim);

    typename xt::xarray<pcf_time_type>::shape_type targetShape;
    targetShape.reserve(in.shape().size());
    for (auto d : (*inBegin).shape())
    {
      targetShape.push_back(d);
    }

    xt::xarray<pcf_time_type> ret(targetShape);

    auto retFlat = xt::flatten(ret);
    auto flatShape = retFlat.shape(0);

    auto nAlongAxis = std::distance(inBegin, inEnd);

    std::vector<pcf_time_type> timesAlongAxis;
    timesAlongAxis.resize(nAlongAxis);
    for (size_t i = 0; i < flatShape; ++i)
    {
      size_t j = 0;
      for (auto it = inBegin; it != inEnd; ++it)
      {
        auto flat = xt::flatten(*it);
        timesAlongAxis[j] = op.get_time_of_interest(flat[j]);
        ++j;
      }

      retFlat[i] = std::reduce(timesAlongAxis.begin(), timesAlongAxis.end(), timesAlongAxis[0], op);
    }

    return ret;
  }


  template <typename XArrayT, typename XExpressionT>
  XArrayT parallel_matrix_reduce(const XExpressionT& in, size_t dim, Executor& exec = default_executor())
  {
    using pcf_type = typename XArrayT::value_type;
    using pcf_value_type = typename pcf_type::value_type;

    auto inBegin = xt::axis_begin(in, dim);
    auto inEnd = xt::axis_end(in, dim);

    auto nAlongAxis = std::distance(inBegin, inEnd);

    XArrayT ret = xt::zeros_like(*inBegin);

    auto retFlat = xt::flatten(ret);
    auto flatShape = retFlat.shape(0);

    // TODO: this is bad!
    std::vector<pcf_type> fs;

    for (size_t i = 0; i < flatShape; ++i)
    {
      fs.clear();
      fs.resize(nAlongAxis);
      auto j = size_t(0);
      for (auto it = inBegin; it != inEnd; ++it)
      {
        auto flat = xt::flatten(*it);
        fs.at(j) = flat[{i}];
        ++j;
      }
      retFlat[{i}] = parallel_reduce(fs.begin(), fs.end(), [](const typename pcf_type::rectangle_type& rect) {
        return rect.top + rect.bottom;
        }, /* TODO: This isn't very nice... */ 8, exec);
    }

    for (size_t i = 0; i < retFlat.shape()[{0}]; ++i)
    {
      retFlat[{i}] /= static_cast<pcf_value_type>(std::distance(xt::axis_begin(in, dim), xt::axis_end(in, dim)));
    }

    return ret;
  }
}

#endif
