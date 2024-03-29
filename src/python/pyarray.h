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

#include <mpcf/pcf.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xlayout.hpp>

#include <vector>

namespace mpcf_py
{

  class Shape
  {
  public:
    Shape(const std::vector<size_t>& data)
      : m_data(data)
    { }

    Shape(std::vector<size_t>&& data)
      : m_data(std::move(data))
    { }

    const std::vector<size_t>& data() const
    {
      return m_data;
    }

    size_t at(size_t i) const
    {
      return m_data.at(i);
    }

    size_t size() const
    {
      return m_data.size();
    }

    auto begin() const
    {
      return m_data.begin();
    }

    auto end() const
    {
      return m_data.end();
    }

  private:
    std::vector<size_t> m_data;
  };

  class Index
  {
  public:
    Index(const std::vector<size_t>& data)
      : m_data(data)
    { }

    Index(std::vector<size_t>&& data)
      : m_data(std::move(data))
    { }

    const std::vector<size_t>& data() const
    {
      return m_data;
    }

    size_t at(size_t i) const
    {
      return m_data.at(i);
    }

    size_t size() const
    {
      return m_data.size();
    }

    auto begin() const
    {
      return m_data.begin();
    }

    auto end() const
    {
      return m_data.end();
    }

  private:
    std::vector<size_t> m_data;
  };

  struct StridedSliceVector
  {
    xt::xstrided_slice_vector data;
  };

  namespace detail
  {
    // This function is purely for type inference in decltype
    template <typename XArrayT>
    auto get_view_helper(XArrayT&& arr)
    {
      const xt::xstrided_slice_vector sv;
      return xt::strided_view(arr, sv);
    }

    template <typename xarray_type>
    using xstrided_view = decltype(detail::get_view_helper<xarray_type>(std::declval<xarray_type>()));

    template <typename xarray_type>
    using xstrided_view_view = decltype(detail::get_view_helper<xstrided_view<xarray_type>>(std::declval<xstrided_view<xarray_type>>()));
  }

  namespace detail
  {
    template <typename ArrayT>
    inline typename ArrayT::xshape_type to_xshape(const Shape& in)
    {
      typename ArrayT::xshape_type s;
      s.resize(in.size());
      std::copy(in.data().begin(), in.data().end(), s.begin());
      return s;
    }

    template <typename ArrayT>
    inline Shape to_Shape(const typename ArrayT::xshape_type& in)
    {
      std::vector<size_t> s;
      s.resize(in.size());
      std::copy(in.begin(), in.end(), s.begin());
      return Shape(std::move(s));
    }
  }

  template <typename ArrayT>
  class StridedView
  {
  public:
    using self_type = StridedView;
    using xshape_type = typename ArrayT::xshape_type;
    using xarray_type = typename ArrayT::xarray_type;
    using xview_type = typename ArrayT::xstrided_view_type;
    using value_type = typename ArrayT::value_type;
    using xstrided_view_type = detail::xstrided_view<xview_type>;

    StridedView(xview_type data)
      : m_data(data)
    { }

    Shape shape() const
    {
      return detail::to_Shape<ArrayT>(m_data.shape());
    }

    StridedView<self_type> view(const StridedSliceVector& sv)
    {
      xstrided_view_type view = get_xview(sv);
      return StridedView<self_type>(view);
    }

    void assign_element(const Index& index, const value_type& val)
    {
      m_data.element(index.begin(), index.end()) = val;
    }

    const value_type& get_element(const Index& index) const
    {
      return m_data.element(index.begin(), index.end());
    }

  private:
    auto get_xview(const StridedSliceVector& sv)
    {
      return xt::strided_view(m_data, sv.data);
    }

    xview_type m_data;
  };

  template <typename Tt, typename Tv>
  class NdArray
  {
  public:
    using self_type = NdArray;
    using value_type = mpcf::Pcf<Tt, Tv>;

    using xarray_type = xt::xarray<value_type>;
    using xshape_type = typename xarray_type::shape_type;

    using xstrided_view_type = detail::xstrided_view<xarray_type>;

    static NdArray make_zeros(const Shape& shape)
    {
      NdArray arr;
      arr.m_data = xarray_type(detail::to_xshape<self_type>(shape));
      return arr;
    }

    Shape shape() const
    {
      return detail::to_Shape<self_type>(m_data.shape());
    }

    StridedView<self_type> view(const StridedSliceVector& sv)
    {
      return StridedView<self_type>(get_xview(sv));
    }

#if 0
    StridedView<StridedView<self_type>> view(const StridedSliceVector& sv)
    {
      auto view = get_xview(sv);
      xt::xstrided_slice_vector svInner;
      for (auto _ : view.shape())
      {
        svInner.emplace_back(xt::all());
      }

      auto viewInner = xt::strided_view(view, svInner);
      return StridedView<StridedView<self_type>>(viewInner);
    }
#endif

    xarray_type& data() { return m_data; }
    const xarray_type& data() const { return m_data; }

    value_type& at(const std::vector<size_t>& pos)
    {
      return m_data[pos];
    }

  private:
    auto get_xview(const StridedSliceVector& sv)
    {
      return xt::strided_view(m_data, sv.data);
    }

    xarray_type m_data;
  };

}
