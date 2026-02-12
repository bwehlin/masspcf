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
#include <algorithm>

#include <iostream>
#include <optional>

#include "config.h"

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

    enum class ViewType
    {
      Base,           // Normal indexing (no reshaping, etc.)
      Flattened       // Flattened view (1-d indexing)
    };

    Tensor(const std::vector<size_t>& shape, const T& init = {});
    Tensor() : Tensor({}, {}) { }

    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return m_strides; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] size_t offset() const noexcept { return m_offset; }
    [[nodiscard]] value_type* data() const noexcept { return m_data.get(); }

    template <typename SliceVector>
    [[nodiscard]] Tensor operator[](SliceVector sliceVector) const;

    [[nodiscard]] Tensor operator[](std::initializer_list<ptrdiff_t> sliceList) const;

    [[nodiscard]] const T& _get_element(const std::vector<size_t>& index) const;

    void _set_element(const std::vector<size_t>& index, const T& val);

    Tensor flatten() const;

    template <typename F>
    void apply(F&& f);

    class AxisIterator
    {
    public:
      AxisIterator(Tensor* tensor, size_t dim, ptrdiff_t pos)
        : m_tensor(tensor), m_dim(dim)
      {
        m_sliceVector.resize(tensor->shape().size(), SliceAll());
        m_sliceVector[dim] = SliceIndex{ pos };
      }

      Tensor operator*() const
      {
        return (*m_tensor)[m_sliceVector];
      }

      AxisIterator& operator++()
      {
        ++std::get<SliceIndex>(m_sliceVector[m_dim]).index;
        return *this;
      }

    private:
      std::vector<Slice> m_sliceVector;
      Tensor* m_tensor;
      size_t m_dim = 0_uz;
    };

  private:
    template <typename SliceVector>
    [[nodiscard]] Tensor extract(SliceVector sliceVector) const;

    [[nodiscard]] size_t get_total_size() const;

    [[nodiscard]] size_t index_to_data_index(const std::vector<size_t>& index) const;
    [[nodiscard]] const T& index_to_ref(const std::vector<size_t>& index) const;
    [[nodiscard]] T& index_to_ref(const std::vector<size_t>& index);

    std::vector<size_t> m_strides;
    std::vector<size_t> m_shape;
    std::shared_ptr<value_type[]> m_data;
    size_t m_offset = 0ul;

    ViewType m_viewType = ViewType::Base;
    bool m_isContiguous = true;
  };

}

#include "tensor.tpp"

#endif //MASSPCF_TENSOR_H
