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

#ifndef MASSPCF_TENSOR_H
#define MASSPCF_TENSOR_H

#include <memory>
#include <vector>
#include <variant>
#include <numeric>

#include "config.h"

#include <pybind11/stl.h>

namespace mpcf
{

  struct SliceAll { };
  struct SliceRange
  {
    ptrdiff_t start = 0;
    ptrdiff_t end = 0;
  };

  [[nodiscard]] inline auto all() { return SliceAll(); }
  [[nodiscard]] inline auto range(ptrdiff_t start, ptrdiff_t end)
  {
    return SliceRange{ .start = start, .end = end };
  }

  using Slice = std::variant<SliceAll, SliceRange>;

  template <typename T>
  class Tensor
  {
  public:

    using value_type = T;

    Tensor(const std::vector<size_t>& shape, const T& init = {})
      : m_shape(shape)
    {
      auto sz = get_total_size();
      m_data = std::make_shared<T[]>(sz);
      std::fill_n(m_data.get(), sz, init);

      // Compute strides
      m_strides.resize(m_shape.size());
      std::partial_sum(m_shape.begin(), m_shape.end(), m_strides.rbegin(), std::multiplies<>());
    }

    Tensor() : Tensor({}) { }

    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return m_strides; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] size_t offset() const noexcept { return m_offset; }
    [[nodiscard]] value_type* data() const noexcept { return m_data.get(); }

#if 0
    template <typename SliceVector>
    [[nodiscard]] Tensor operator[](const SliceVector& sliceVector) const
    {

    }
#endif

  private:
    [[nodiscard]] size_t get_total_size() const
    {
      return std::accumulate(m_shape.begin(), m_shape.end(), 0_uz, std::plus<>());
    }

    std::vector<size_t> m_strides;
    std::vector<size_t> m_shape;
    std::shared_ptr<value_type[]> m_data;
    size_t m_offset = 0ul;

  };

}

#endif //MASSPCF_TENSOR_H
