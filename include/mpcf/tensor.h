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

#include <iostream>
#include <optional>

#include "config.h"

#include <pybind11/stl.h>

namespace mpcf
{

  struct SliceAll { };

  struct SliceIndex
  {
    ptrdiff_t index;
  };

  struct SliceRange
  {
    std::optional<ptrdiff_t> start;
    std::optional<ptrdiff_t> stop;
    std::optional<ptrdiff_t> step;
  };

  using Slice = std::variant<SliceIndex, SliceRange>;

  [[nodiscard]] inline Slice all() { return Slice{SliceRange{}}; }

  [[nodiscard]] inline Slice index(ptrdiff_t index)
  {
    return Slice{SliceIndex{ .index = index }};
  }

  [[nodiscard]] inline Slice range(std::optional<ptrdiff_t> start, std::optional<ptrdiff_t> stop, std::optional<ptrdiff_t> step)
  {
    return Slice{SliceRange{ .start = start, .stop = stop, .step = step }};
  }

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
      if (!m_shape.empty())
      {
        m_strides = m_shape;

        std::partial_sum(m_shape.rbegin(), std::prev(m_shape.rend()), std::next(m_strides.rbegin()), std::multiplies<>());
        m_strides.back() = 1;
      }
    }

    Tensor() : Tensor({}, {}) { }

    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return m_strides; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] size_t offset() const noexcept { return m_offset; }
    [[nodiscard]] value_type* data() const noexcept { return m_data.get(); }

    template <typename SliceVector>
    [[nodiscard]] Tensor operator[](SliceVector sliceVector) const
    {
      Tensor ret;

      ret.m_data = m_data; // view

      // temporary just to get sizes
      ret.m_shape = m_shape;
      ret.m_strides = m_strides;

      size_t i = 0;
      for (auto & slice : sliceVector)
      {
        std::visit([i, &slice, &ret, this](auto&& arg) {
          using argT = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<argT, SliceIndex>)
          {
            ret.m_shape[i] = 1;
            ret.m_offset += arg.index * ret.m_strides[i];
            std::cout << "i " << i << " offset += " << arg.index * ret.m_strides[i] << std::endl;
          }
          else if constexpr (std::is_same_v<argT, SliceRange>)
          {
            if (!arg.start)
            {
              arg.start = 0_z;
            }
            if (!arg.stop)
            {
              arg.stop = static_cast<ptrdiff_t>(ret.m_shape[i]);
            }
            if (!arg.step)
            {
              arg.step = 1_z;
            }

            // TODO: step
            ret.m_shape[i] = (*arg.stop - *arg.start) / *arg.step;
            std::cout << "sta " << *arg.start << " ste " << *arg.step << " sto " << *arg.stop <<  " r" << *arg.stop - *arg.start << std::endl;
            ret.m_strides[i] *= *arg.step;
            //for ()
            ret.m_offset += *arg.start * ret.m_strides[i];
            //std::cout << "i " << i << " offset += " << *arg.start * ret.m_strides[i] << std::endl;
          }
          // For SliceAll, don't modify shape
        }, slice);
        ++i;
      }


      return ret;
    }

    [[nodiscard]] const T& _get_element(const std::vector<size_t>& index) const
    {
      return index_to_ref(index);
    }

    void _set_element(const std::vector<size_t>& index, const T& val)
    {
      index_to_ref(index) = val;
    }

  private:
    [[nodiscard]] size_t get_total_size() const
    {
      return std::accumulate(m_shape.begin(), m_shape.end(), 1_uz, std::multiplies<>());
    }

    [[nodiscard]] size_t index_to_data_index(const std::vector<size_t>& index) const
    {
      auto ret = std::inner_product(index.begin(), index.end(), m_strides.begin(), 0_uz);
      ret += m_offset;
      //std::cout << "Translated " <<  " -> " << ret << std::endl;
      return ret;
    }

    [[nodiscard]] const T& index_to_ref(const std::vector<size_t>& index) const
    {
      return m_data[index_to_data_index(index)];
    }

    [[nodiscard]] T& index_to_ref(const std::vector<size_t>& index)
    {
      return m_data[index_to_data_index(index)];
    }

    std::vector<size_t> m_strides;
    std::vector<size_t> m_shape;
    std::shared_ptr<value_type[]> m_data;
    size_t m_offset = 0ul;

  };

}

#endif //MASSPCF_TENSOR_H
