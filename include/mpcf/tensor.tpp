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
    if (m_shape != rhs.shape())
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
    if (m_shape != rhs.shape())
    {
      return true;
    }

    return any_of_idx([this, &rhs](const std::vector<size_t>& idx) {
      return (*this)(idx) != rhs(idx);
    });
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
  template <typename U>
  requires CanDivideTo<T, T, U>
  Tensor<T>& Tensor<T>::operator/=(const U& u)
  {
    apply([&u](T& val){
      val /= u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanDivideTo<T, T, U>
  Tensor<T> Tensor<T>::operator/(const U& u) const
  {
    Tensor<T> ret = copy();
    ret /= u;
    return ret;
  }

  template <typename T>
  template <typename U>
  requires CanMultiplyTo<T, T, U>
  Tensor<T>& Tensor<T>::operator*=(const U& u)
  {
    apply([&u](T& val){
      val *= u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanMultiplyTo<T, T, U>
  Tensor<T> Tensor<T>::operator*(const U& u) const
  {
    Tensor<T> ret = copy();
    ret *= u;
    return ret;
  }

  template <typename T>
  template <typename U>
  requires CanAddTo<T, T, U>
  Tensor<T>& Tensor<T>::operator+=(const U& u)
  {
    apply([&u](T& val){
      val += u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanAddTo<T, T, U>
  Tensor<T> Tensor<T>::operator+(const U& u) const
  {
    Tensor<T> ret = copy();
    ret += u;
    return ret;
  }

  template <typename T>
  template <typename U>
  requires CanSubtractTo<T, T, U>
  Tensor<T>& Tensor<T>::operator-=(const U& u)
  {
    apply([&u](T& val){
      val -= u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanSubtractTo<T, T, U>
  Tensor<T> Tensor<T>::operator-(const U& u) const
  {
    Tensor<T> ret = copy();
    ret -= u;
    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::operator-() const requires CanNegate<T>
  {
    Tensor<T> ret = copy();
    ret.apply([](T& val){
      val = -val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanMultiplyTo<T, U, T>
  Tensor<T> operator*(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u * val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanAddTo<T, U, T>
  Tensor<T> operator+(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u + val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanSubtractTo<T, U, T>
  Tensor<T> operator-(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u - val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanDivideTo<T, U, T>
  Tensor<T> operator/(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u / val;
    });
    return ret;
  }

  /**
   * Elementwise power for tensors.
   *
   * Returns a new tensor whose elements are each raised to `exponent`
   * via `mpcf::pow`. Works for scalar element types (float, double)
   * and any type that provides a `.pow()` member (e.g. Pcf).
   *
   * @param t        the input tensor
   * @param exponent the exponent to raise each element to
   * @return a new tensor with transformed elements
   */
  template <typename T, typename U>
  requires CanPow<T, U>
  Tensor<T> pow(const Tensor<T>& t, U exponent)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&exponent](T& val){
      val = mpcf::pow(val, exponent);
    });
    return ret;
  }

  /**
   * In-place elementwise power for tensors.
   *
   * Raises each element of `t` to `exponent` in place via `mpcf::pow`.
   *
   * @param t        the tensor to modify
   * @param exponent the exponent to raise each element to
   */
  template <typename T, typename U>
  requires CanPow<T, U>
  void ipow(Tensor<T>& t, U exponent)
  {
    t.apply([&exponent](T& val){
      val = mpcf::pow(val, exponent);
    });
  }

  // ============================================================================
  // Broadcasting
  // ============================================================================

  template <typename T>
  Tensor<T> Tensor<T>::broadcast_to(const std::vector<size_t>& target_shape) const
  {
    size_t ndim = target_shape.size();
    size_t src_ndim = m_shape.size();

    if (ndim < src_ndim)
    {
      throw std::invalid_argument("Cannot broadcast shape " +
          shape_to_string(m_shape) + " to " + shape_to_string(target_shape) +
          ": target has fewer dimensions");
    }

    Tensor ret;
    ret.m_data = m_data;
    ret.m_offset = m_offset;
    ret.m_shape = target_shape;
    ret.m_strides.resize(ndim);
    ret.m_isContiguous = false;

    size_t prepended = ndim - src_ndim;
    for (size_t i = 0; i < ndim; ++i)
    {
      if (i < prepended)
      {
        ret.m_strides[i] = 0;
      }
      else
      {
        size_t si = i - prepended;
        if (m_shape[si] == target_shape[i])
        {
          ret.m_strides[i] = m_strides[si];
        }
        else if (m_shape[si] == 1)
        {
          ret.m_strides[i] = 0;
        }
        else
        {
          throw std::invalid_argument("Cannot broadcast shape " +
              shape_to_string(m_shape) + " to " + shape_to_string(target_shape));
        }
      }
    }

    return ret;
  }

  // Tensor-Tensor arithmetic with broadcasting

  namespace detail
  {
    template <typename T, typename BinaryOp>
    Tensor<T> broadcast_binop(const Tensor<T>& lhs, const Tensor<T>& rhs, BinaryOp op)
    {
      auto out_shape = broadcast_shapes(lhs.shape(), rhs.shape());
      auto lhs_view = lhs.broadcast_to(out_shape);
      auto rhs_view = rhs.broadcast_to(out_shape);
      Tensor<T> result(out_shape);
      result.walk([&](const std::vector<size_t>& idx) {
        result(idx) = op(lhs_view(idx), rhs_view(idx));
      });
      return result;
    }

    template <typename T, typename BinaryOp>
    Tensor<T>& broadcast_binop_inplace(Tensor<T>& lhs, const Tensor<T>& rhs, BinaryOp op)
    {
      auto out_shape = broadcast_shapes(lhs.shape(), rhs.shape());
      if (out_shape != lhs.shape())
        throw std::invalid_argument("Cannot broadcast in-place: output shape " +
            shape_to_string(out_shape) + " differs from LHS shape " + shape_to_string(lhs.shape()));
      auto rhs_view = rhs.broadcast_to(out_shape);
      lhs.walk([&](const std::vector<size_t>& idx) {
        lhs(idx) = op(lhs(idx), rhs_view(idx));
      });
      return lhs;
    }
  }

  template <typename T>
  Tensor<T> Tensor<T>::operator+(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b){ return a + b; }); }

  template <typename T>
  Tensor<T> Tensor<T>::operator-(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b){ return a - b; }); }

  template <typename T>
  Tensor<T> Tensor<T>::operator*(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b){ return a * b; }); }

  template <typename T>
  Tensor<T> Tensor<T>::operator/(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b){ return a / b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator+=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a + b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator-=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a - b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator*=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a * b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator/=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a / b; }); }

  template <typename T>
  [[nodiscard]] size_t Tensor<T>::size() const noexcept
  {
    if (m_shape.empty())
    {
      return 0_uz;
    }
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
    Tensor ret;
    if (m_isContiguous)
    {
      ret = *this;
    }
    else
    {
      ret = copy();
    }

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
    mpcf::walk(*this, std::forward<UnaryFunc>(f));
  }

  template <typename T>
  template <typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, const T&>
#endif
  bool Tensor<T>::any_of(UnaryFunc&& f) const
  {
    return any_of_idx([this, &f](const std::vector<size_t>& idx){
      return f((*this)(idx));
    });
  }

  template <typename T>
  template <typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  bool Tensor<T>::any_of_idx(UnaryFunc&& f) const
  {
    bool match = false;

    // This could be made more efficient
    walk([this, &match, &f](const std::vector<size_t>& idx) {

      if (f(idx))
      {
        match = true;
        return false;
      }

      return true;
    });

    return match;

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

    // For now, we'll drop the assumption that the tensor is contiguous in memory
    // as soon as we extract a subtensor. There are, however, some cases where the resulting subtensor would be
    // contiguous that we can try to optimize for in the future (e.g., extracting the top n rows of a matrix).
    // Things will work fine with this assumption dropped but certain operations could be a little slower (probably
    // unlikely to matter for the type of things we're targeting).
    ret.m_isContiguous = false;

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
            start = 0;
          if (stop > static_cast<std::ptrdiff_t>(ret.m_shape[i]))
            stop = static_cast<std::ptrdiff_t>(ret.m_shape[i]);

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
            else
            {
              ret.m_shape[i] = (stop - start + step - 1_z) / step;
            }
          }
          else
          {
            throw std::runtime_error("Negative step not supported...");
          }

          ret.m_offset += start * ret.m_strides[i]; // use clamped start
          ret.m_strides[i] *= step;

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

  template <IsTensor TTensor, typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  void walk(const TTensor& tensor, UnaryFunc&& f)
  {
    auto shape_range = tensor.shape();
    std::vector<size_t> shape(std::begin(shape_range), std::end(shape_range));

    if (shape.empty() || std::any_of(shape.begin(), shape.end(), [](size_t n){ return n == 0; }))
    {
      return;
    }

    auto ndim = shape.size();
    std::vector<size_t> cur(ndim, 0_uz);

    while (true)
    {
      if constexpr (std::is_same_v<decltype(f(cur)), bool>)
      {
        if (!f(cur))
        {
          return;
        }
      }
      else
      {
        f(cur);
      }

      for (ptrdiff_t i = ndim - 1; i >= 0; --i)
      {
        ++cur[i];

        if (cur[i] < shape[i])
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
}
