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

#include <taskflow/algorithm/for_each.hpp>

#include "../pcf.h"
#include "../executor.h"
#include "reduce.h"

namespace mpcf
{
  namespace detail
  {
    template <typename XArrayT>
    class SliceBasePointIterator
    {
    public:
      using value_type = typename XArrayT::value_type;
      using difference_type = std::ptrdiff_t;
      using iterator_category = std::forward_iterator_tag;



    };
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

#if 1

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
        });
    }

#endif

#if 0

    for (auto it = inBegin; it != inEnd; ++it)
    {
      auto flat = xt::flatten(*it);
      for (auto i = 0; i < flatShape; ++i)
      {
        retFlat[{i}] += flat.at(i);
      }
    }
#endif
    
#if 0
    for (auto it = inBegin; it != inEnd; ++it)
    {
      tf::Taskflow flow;

      auto flat = xt::flatten(*it);
      flow.for_each_index(size_t(0), size_t(flatShape), size_t(1), [&retFlat, &flat](size_t i) {
        retFlat[{i}] += flat.at(i);
        });



      exec.cpu()->run(std::move(flow)).wait();
    }

#endif

    for (size_t i = 0; i < retFlat.shape()[{0}]; ++i)
    {
      retFlat[{i}] /= static_cast<pcf_value_type>(std::distance(xt::axis_begin(in, dim), xt::axis_end(in, dim)));
    }

    return ret;
  }
}

#endif
