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

#include <type_traits>
#include <variant>
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

    template <typename XShapeT>
    inline Shape to_Shape2(const XShapeT& in)
    {
      std::vector<size_t> s;
      s.resize(in.size());
      std::copy(in.begin(), in.end(), s.begin());
      return Shape(std::move(s));
    }
  }

  namespace detail
  {
    template <typename T>
    std::remove_pointer_t<T>* ptr(T& v)
    {
      if constexpr (std::is_pointer_v<T>)
      {
        return v;
      }
      else
      {
        return &v;
      }
    }

    template <typename T>
    std::remove_pointer_t<T>& ref(T& v)
    {
      if constexpr (std::is_pointer_v<T>)
      {
        return *v;
      }
      else
      {
        return v;
      }
    }

    template <typename T>
    const std::remove_pointer_t<T>& cref(T& v)
    {
      if constexpr (std::is_pointer_v<T>)
      {
        return *v;
      }
      else
      {
        return v;
      }
    }
  }

  template <typename ArrayT>
  class View
  {
  public:
    using self_type = View;
    using array_type = ArrayT;
    using value_type = typename array_type::value_type;
    using xarray_type = typename array_type::xarray_type;

  private:
    using xv = detail::xstrided_view<xarray_type>;
    template <typename T> using xvv = detail::xstrided_view<T>;

    using xvend = xvv<xvv<xvv<xvv<xv>>>>;

  public:
  
    template <typename T>
    static View<ArrayT> create(T xview)
    {
      View<ArrayT> view;
      view.m_data = xview;
      return view;
    }

    View strided_view(const StridedSliceVector& sv)
    {
      return std::visit([&sv](auto&& arg) -> View
        {
          if constexpr (!std::is_same_v<std::decay_t<decltype(arg)>, std::monostate>  && !std::is_same_v<std::decay_t<decltype(arg)>, xvend > )
          {
            return View<ArrayT>::create(
              xt::strided_view(*detail::ptr(arg), sv.data));
          }
          else
          {
            throw std::runtime_error("Unsupported operation on this type of view.");
          }
        }, m_data);
    }

    View transpose()
    {
      return std::visit([](auto&& arg) -> View
        {
          if constexpr (!std::is_same_v<std::decay_t<decltype(arg)>, std::monostate> && !std::is_same_v<std::decay_t<decltype(arg)>, xvend >)
          {
            return View<ArrayT>::create(
              xt::transpose(*detail::ptr(arg)));
          }
          else
          {
            throw std::runtime_error("Unsupported operation on this type of view.");
          }
        }, m_data);
    }

    Shape get_shape() const
    {
      return std::visit([this](auto&& arg) -> Shape
        {
          if constexpr (!std::is_same_v<std::decay_t<decltype(arg)>, std::monostate>)
          {
            return detail::to_Shape2((detail::ptr(arg))->shape());
          }
          else
          {
            throw std::runtime_error("Unsupported operation on this type of view.");
          }
        }, m_data);
    }

    void assign(const View& from)
    {
      std::visit([this, &from](auto&& toArg) {

          if constexpr (!std::is_same_v<std::decay_t<decltype(toArg)>, std::monostate>)
          {
            std::visit([&toArg](auto&& fromArg) {

              if constexpr (!std::is_same_v<std::decay_t<decltype(fromArg)>, std::monostate>)
              {
                detail::ref(toArg) = detail::cref(fromArg);
              }
              else
              {
                throw std::runtime_error("Unsupported operation on this type of view (from view).");
              }

              }, from.m_data);
          }
          else
          {
            throw std::runtime_error("Unsupported operation on this type of view (to view).");
          }

        }, m_data);
    }

    value_type& at(const std::vector<size_t>& pos)
    {
      return std::visit([this, &pos](auto&& arg) -> value_type&
        {
          if constexpr (!std::is_same_v<std::decay_t<decltype(arg)>, std::monostate>)
          {
            return detail::ref(arg)[pos];
          }
          else
          {
            throw std::runtime_error("Unsupported operation on this type of view.");
          }
        }, m_data);
    }

  private:

    std::variant<
      std::monostate,
      xarray_type*,
      xv, xvv<xv>, xvv<xvv<xv>>, xvv<xvv<xvv<xv>>>, xvv<xvv<xvv<xvv<xv>>>>
    > m_data;
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
    
    View<self_type> as_view()
    {
      return View<self_type>::create(&m_data);
    }

    View<self_type> strided_view(const StridedSliceVector& sv)
    {
      auto xview = get_xview(sv);
      return View<self_type>::create(xview);
    }

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
