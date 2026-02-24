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

#include "../pcf.h"
#include "../tensor.h"
#include "../executor.h"
#include "reduce.h"

#include <iostream>

namespace mpcf
{
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
    auto shape = in.shape();

    std::vector<size_t> inIdx(shape.size(), 0_uz);

    auto inDimSize = shape[dim];

    shape.erase(shape.begin() + dim);
    if (shape.empty())
    {
      shape.resize(1, 1);
    }

    Tensor<PcfT> ret(shape);
    ret.walk([&ret, &in, &inIdx, inDimSize, dim](const std::vector<size_t>& idx){

      std::copy(idx.begin(), idx.begin() + dim, inIdx.begin());
      if (inIdx.size() > 1)
      {
        // MSVC debug does not like (inIdx.begin() + dim + 1) even if nothing is written there it seems.
        std::copy(idx.begin() + dim, idx.end(), inIdx.begin() + dim + 1);
      }

      std::vector<PcfT> tmp; // TODO: Rewrite without need to copy
      tmp.reserve(inDimSize);
      for (auto i = 0_uz; i < inDimSize; ++i)
      {
        inIdx[dim] = i;
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

  template <typename PcfT, typename UnaryF, typename MaxOp>
  auto max_element(const Tensor<PcfT>& in, size_t dim, UnaryF&& f, MaxOp&& maxOp, Executor& exec = default_executor())
  {
    using OutTensorT = Tensor<std::decay_t<
        decltype(f(std::declval<PcfT>()))
    >>;

    using OutValueT = typename OutTensorT::value_type;

    static_assert(std::invocable<MaxOp, OutValueT, OutValueT>);

    auto shape = in.shape();

    std::vector<size_t> inIdx(shape.size(), 0_uz);

    auto inDimSize = shape[dim];

    shape.erase(shape.begin() + dim);
    if (shape.empty())
    {
      shape.resize(1, 1);
    }
    OutTensorT ret(shape);

    //tf::Taskflow

    ret.walk([&ret, &in, &inIdx, inDimSize, dim, &f, &maxOp](const std::vector<size_t>& idx){

      std::copy(idx.begin(), idx.begin() + dim, inIdx.begin());
      if (inIdx.size() > 1)
      {
        // MSVC debug does not like (inIdx.begin() + dim + 1) even if nothing is written there it seems.
        std::copy(idx.begin() + dim, idx.end(), inIdx.begin() + dim + 1);
      }

      std::vector<PcfT> tmp; // TODO: Rewrite without need to copy
      tmp.reserve(inDimSize);
      for (auto i = 0_uz; i < inDimSize; ++i)
      {
        inIdx[dim] = i;
        tmp.emplace_back( in(inIdx) );
      }

      auto & out = ret(idx);

      auto init = f(*tmp.begin());
      out =  std::transform_reduce(tmp.begin() + 1, tmp.end(), init, maxOp, f);

    });

    return ret;
  }
}

#endif
