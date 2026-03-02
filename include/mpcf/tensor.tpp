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

#include <functional>

namespace mpcf
{
  template <typename T>
  Tensor<T>::Tensor(const std::vector<size_t>& shape, const T& init)
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

  template <typename T>
  Tensor<T>& Tensor<T>::operator=(const T& val)
  {
    apply([&val](T& element){ element = val; });
    return *this;
  }

  template <typename T>
  template <typename U> requires std::equality_comparable_with<U, T>
  bool Tensor<T>::operator==(const Tensor<U>& rhs) const
  {
    if (m_shape != rhs.m_shape)
    {
      return false;
    }

    bool equal = true;
    walk([&equal, this, &rhs](const std::vector<size_t>& idx) {
      return (equal &= ( (*this)(idx) == rhs(idx) ) );
    });

    return equal;
  }

  template <typename T>
  template <typename U> requires std::equality_comparable_with<U, T>
  bool Tensor<T>::operator!=(const Tensor<U>& rhs) const
  {
    if (m_shape != rhs.m_shape)
    {
      return false;
    }

    bool equal = true;
    walk([&equal, this, &rhs](const std::vector<size_t>& idx) {
      return !(equal &= ( (*this)(idx) == rhs(idx) ) );
    });

    return !equal;
  }

  template <typename T>
  template <typename SliceVector>
  Tensor<T> Tensor<T>::operator[](SliceVector sliceVector) const
  {
    return extract(sliceVector);
  }

  template <typename T>
  const T& Tensor<T>::operator()(const std::vector<size_t>& index) const
  {
    return index_to_ref(index);
  }

  template <typename T>
  T& Tensor<T>::operator()(const std::vector<size_t>& index)
  {
    return index_to_ref(index);
  }

  template <typename T>
  const T& Tensor<T>::operator()(size_t index) const
  {
    return index_to_ref({ index });
  }

  template <typename T>
  T& Tensor<T>::operator()(size_t index)
  {
    return index_to_ref({ index });
  }

  template <typename T>
  template <typename U>
  requires std::is_convertible_v<U, T>
  void Tensor<T>::assign_from(const Tensor<U>& rhs)
  {
    if (shape() != rhs.shape())
    {
      throw std::runtime_error("Incommensurate shapes (tried to assign from a tensor of shape " + shape_to_string(rhs.shape()) + " to a tensor of shape " +
          shape_to_string(shape()) + ")");
    }

    walk([this, &rhs](const std::vector<size_t>& idx){
      (*this)(idx) = rhs(idx);
    });
  }

  template <typename T>
  [[nodiscard]] size_t Tensor<T>::size() const noexcept
  {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1_uz, std::multiplies<size_t>());
  }

  template <typename T>
  Tensor<T> Tensor<T>::copy() const
  {
    Tensor<T> ret(shape());

    walk([&ret, this](const std::vector<size_t>& idx){
      ret(idx) = (*this)(idx);
    });

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::flatten() const
  {
    if (!m_isContiguous)
    {
      throw std::runtime_error("flatten() is only available for contiguous tensors in this release (please file an issue if you need this for your case).");
    }

    Tensor ret = *this;
    ret.m_viewType = ViewType::Flattened;
    ret.m_shape = { get_total_size() };
    ret.m_strides = { 0_uz };
    return ret;
  }

  template <typename T>
  template <typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  void Tensor<T>::walk(UnaryFunc&& f) const
  {
    if (m_shape.empty() || std::any_of(m_shape.begin(), m_shape.end(), [](size_t n){ return n == 0; }))
    {
      return;
    }
    auto ndim = m_shape.size();

    std::vector<size_t> cur(ndim, 0_uz);

    while (true)
    {
      if constexpr (std::is_same_v<decltype(f(cur)), bool>)
      {
        // If f returns bool, stop walking on false being returned.
        if (!f(cur))
        {
          return;
        }
      }
      else
      {
        f(cur);
      }


      for (ptrdiff_t i = ndim - 1; i >= 0 ; --i)
      {
        ++cur[i];

        if (cur[i] < m_shape[i])
        {
          break;
        }

        if (i == 0)
        {
          return;
        }

        cur[i] = 0;
      }
    }
  }

  template <typename T>
  template <typename UnaryFunc> requires std::invocable<UnaryFunc, T&>
  void Tensor<T>::apply(UnaryFunc&& f)
  {
    walk([&f, this](const std::vector<size_t>& idx){ f((*this)(idx)); });
  }

  template <typename T>
  template <typename SliceVector>
  Tensor<T> Tensor<T>::extract(SliceVector sliceVector) const
  {
    Tensor ret;

    ret.m_data = m_data; // view

    // temporary just to get sizes
    ret.m_shape = m_shape;
    ret.m_strides = m_strides;

    ret.m_offset = m_offset;

    std::vector<size_t> dimsToDrop;

    size_t i = 0;
    for (auto & slice : sliceVector)
    {
      std::visit([i, &slice, &ret, &dimsToDrop](auto&& arg) {
        using argT = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<argT, SliceIndex>)
        {
          ret.m_shape[i] = 1;
          ret.m_offset += arg.index * ret.m_strides[i];
          dimsToDrop.emplace_back(i);
        }
        else if constexpr (std::is_same_v<argT, SliceRange>)
        {
          // For now, we'll drop the assumption that the tensor is contiguous in memory
          // as soon as we extract a subtensor using ranges. There are, however, some
          // cases where the resulting subtensor would be contiguous that we can try to
          // optimize for in the future (e.g., extracting the top n rows of a matrix).
          // Things will work fine with this assumption dropped but certain operations
          // could be a little slower (probably unlikely to matter for the type of
          // things we're targeting).
          ret.m_isContiguous = false;

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

          auto start = *arg.start;
          auto stop = *arg.stop;
          auto step = *arg.step;

          if (start < 0_z)
          {
            start = 0;
          }

          if (stop > static_cast<std::ptrdiff_t>(ret.m_shape[i]))
          {
            stop = ret.m_shape[i];
          }

          if (step == 0_z)
          {
            ret.m_shape[i] = 0;
          }
          else if (step > 0)
          {
            if (stop <= start)
            {
              ret.m_shape[i] = 0;
            }
            ret.m_shape[i] = (*arg.stop - *arg.start + *arg.step - 1_z) / *arg.step;
          }
          else
          {
            throw std::runtime_error("Negative step not supported in this release (please file an issue if you need this).");
          }


          ret.m_offset += *arg.start * ret.m_strides[i];
          ret.m_strides[i] *= *arg.step;

        }
        // For SliceAll, don't modify shape
      }, slice);
      ++i;
    }

    size_t nDroppedDims = 0_uz;
    for (auto dim : dimsToDrop)
    {
      ret.m_shape.erase(ret.m_shape.begin() + dim - nDroppedDims);
      ret.m_strides.erase(ret.m_strides.begin() + dim - nDroppedDims);
      ++nDroppedDims;
    }

    return ret;
  }


  template <typename T>
  size_t Tensor<T>::get_total_size() const
  {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1_uz, std::multiplies<>());
  }

  template <typename T>
  size_t Tensor<T>::index_to_data_index(const std::vector<size_t>& index) const
  {
    size_t ret = 0_uz;
    switch (m_viewType)
    {
    case ViewType::Base:
      ret = std::inner_product(index.begin(), index.end(), m_strides.begin(), 0_uz);
      ret += m_offset;
      //std::cout << "Translated " <<  " -> " << ret << std::endl;
      return ret;
    case ViewType::Flattened:
      if (index.size() != 1_uz)
      {
        throw std::runtime_error("Index into flat tensor should be 1d.");
      }

      if (m_isContiguous)
      {
        return m_offset + index[0];
      }

      return ret;
    }

    throw std::runtime_error("Unhandled view type!");
  }

  template <typename T>
  const T& Tensor<T>::index_to_ref(const std::vector<size_t>& index) const
  {
    return m_data[index_to_data_index(index)];
  }

  template <typename T>
  T& Tensor<T>::index_to_ref(const std::vector<size_t>& index)
  {
    return m_data[index_to_data_index(index)];
  }
}