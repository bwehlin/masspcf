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

#ifndef MPCF_ALGORITHMS_MATRIX_REDUCE_H
#define MPCF_ALGORITHMS_MATRIX_REDUCE_H

#include <taskflow/algorithm/for_each.hpp>

#include "../pcf.h"
#include "../tensor.h"
#include "../executor.h"
#include "reduce.h"

#include <iostream>

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

#if 0
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
      // Return Shape(1,) array if reducing Shape(n,) (instead of empty shape)
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
#endif

  template <typename T>
  inline void printVec(const char* n, const std::vector<T>& v) {
    std::cout << n << " : ";
    for (auto i : v)
    {
      std::cout << ", " << i;
    }
    std::cout << std::endl;
  };

  template <typename PcfT> //, typename ReductionF>
  Tensor<PcfT> parallel_tensor_reduce(const Tensor<PcfT>& in, size_t dim, Executor& exec = default_executor())
  {
    std::cout << "HELLO" << std::endl;
    auto shape = in.shape();

    std::vector<size_t> inIdx(shape.size(), 0_uz);

    printVec("INSHAPE ", in.shape());
    auto inDimSize = shape[dim];

    shape.erase(shape.begin() + dim);
    if (shape.empty())
    {
      shape.resize(1, 1);
    }
    Tensor<PcfT> ret(shape);
    printVec("RET SHAPE ", ret.shape());


    ret.walk([&ret, &in, &inIdx, inDimSize, dim](const std::vector<size_t>& idx){

      std::copy(idx.begin(), idx.begin() + dim, inIdx.begin());
      if (inIdx.size() > 1)
      {
        // MSVC debug does not like (inIdx.begin() + dim + 1) even if nothing is written there it seems.
        std::copy(idx.begin() + dim, idx.end(), inIdx.begin() + dim + 1);
      }

      printVec("idx", idx);

      std::vector<PcfT> tmp; // TODO: Rewrite without need to copy
      tmp.reserve(inDimSize);
      for (auto i = 0_uz; i < inDimSize; ++i)
      {
        inIdx[dim] = i;
        printVec("  inIdx", inIdx);

        tmp.emplace_back( in(inIdx) );
      }

      auto & out = ret(idx);

      out = reduce(tmp, [inDimSize](const typename PcfT::rectangle_type& rect) {
        return rect.top + rect.bottom;
      });

      out /= inDimSize;
    });

    return ret;
  }
}

#endif
