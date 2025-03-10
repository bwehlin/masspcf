/*
* Copyright 2024-2025 Bjorn Wehlin
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
#include <mpcf/algorithms/matrix_reduce.h>
#include <mpcf/strided_buffer.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xlayout.hpp>

#include <type_traits>
#include <variant>
#include <vector>
#include <initializer_list>
#include <iostream>
#include <sstream>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mpcf_py
{

  class Shape
  {
  public:
    Shape(const std::vector<size_t>& data)
      : m_data(data)
    { }

    Shape(const std::initializer_list<size_t>& data)
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

    std::string to_string() const 
    {
      std::stringstream ss;
      ss << "Shape(";
      for (auto i : m_data)
      {
        ss << i << ", ";
      }
      ss << ")";
      return ss.str();
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

    void append_all()
    {
      data.emplace_back(xt::all());
    }

    void append_range(size_t start, size_t stop, size_t step)
    {
      data.emplace_back(xt::range(start, stop, step));
    }

    void append_range_from(size_t start, size_t step)
    {
      using namespace xt::placeholders;
      data.emplace_back(xt::range(start, _, step));
    }

    void append_range_to(size_t stop, size_t step)
    {
      using namespace xt::placeholders;
      data.emplace_back(xt::range(_, stop, step));
    }
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

    template <typename XShapeT>
    inline Shape to_Shape(const XShapeT& in)
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

  namespace detail
  {
    template <typename T, typename TRet>
    struct throw_unsupported
    {
      [[noreturn]] TRet operator()(T) const
      {
        throw std::runtime_error("Unsupported operation on this type of view.");
      }
    };


    template<class... Ts>
    struct overloaded : Ts... { using Ts::operator()...; };
    template<class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;
  }


  template <typename ArrayT>
  class View
  {
  public:
    using self_type = View;
    using array_type = ArrayT;
    using pcf_type = typename array_type::value_type;
    using time_type = typename pcf_type::time_type;
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
      return std::visit(detail::overloaded {
          [&sv](auto&& arg) -> View { return View<ArrayT>::create(xt::strided_view(*detail::ptr(arg), sv.data)); },
          detail::throw_unsupported<std::monostate, View>(),
          detail::throw_unsupported<xvend, View>()
      }, m_data);
    }

    View transpose()
    {
      return std::visit(detail::overloaded {
          [](auto&& arg) -> View { return View<ArrayT>::create(xt::transpose(*detail::ptr(arg))); },
          detail::throw_unsupported<std::monostate, View>(),
          detail::throw_unsupported<xvend, View>()
      }, m_data);
    }

    Shape get_shape() const
    {
      return std::visit(detail::overloaded {
          [](auto&& arg) -> Shape { return detail::to_Shape((detail::ptr(arg))->shape()); },
          detail::throw_unsupported<std::monostate, Shape>()
      }, m_data);
    }

    void assign(const View& from)
    {
      std::visit(detail::overloaded{
        [&from](auto&& toArg) {
          std::visit(detail::overloaded{
            [&toArg](auto&& fromArg) {
              detail::ref(toArg) = detail::cref(fromArg);
            },
            detail::throw_unsupported<std::monostate, void>()
          }, from.m_data);
        },
        detail::throw_unsupported<std::monostate, void>()
      }, m_data);
    }

    void assign_at(const StridedSliceVector& sv, const pcf_type& f)
    {
      std::visit(detail::overloaded {
          [&sv, &f](auto&& arg) -> void { xt::strided_view(detail::ref(arg), sv.data) = f; },
          detail::throw_unsupported<std::monostate, void>()
      }, m_data);
    }

    pcf_type& at(const std::vector<size_t>& pos)
    {
      return std::visit(detail::overloaded {
          [&pos](auto&& arg) -> pcf_type& { return detail::ref(arg)[pos]; },
          detail::throw_unsupported<std::monostate, pcf_type&>()
      }, m_data);
    }

    array_type reduce_mean(int dim)
    {
      return std::visit(detail::overloaded {
        [dim](auto&& arg) -> array_type { return array_type(std::move(mpcf::parallel_matrix_reduce<xarray_type>(detail::cref(arg), dim))); },
        detail::throw_unsupported<std::monostate, array_type>()
      }, m_data);
    }

    py::array_t<time_type> reduce_max_time(int dim)
    {
      using xtime_type = xt::xarray<time_type>;

      xtime_type timeArr = std::visit(detail::overloaded {
          [dim](auto&& arg) -> xtime_type { return mpcf::matrix_time_reduce<xarray_type>(detail::cref(arg), dim, mpcf::TimeOpMaxTime<pcf_type>()); },
          detail::throw_unsupported<std::monostate, xtime_type>()
      }, m_data);

      py::array_t<time_type> ret(timeArr.shape(), timeArr.strides());
      auto* retData = ret.mutable_data();

      auto flat = xt::flatten(timeArr);
      auto len = flat.shape(0);

      for (size_t i = 0; i < len; ++i)
      {
        retData[i] = flat[{i}];
      }

      return ret;
    }

    pcf_type* buffer()
    {
      return std::visit(detail::overloaded {
        [](auto&& arg) -> pcf_type* { return detail::ptr(arg)->data(); },
        detail::throw_unsupported<std::monostate, pcf_type*>()
      }, m_data);
    }

    size_t offset()
    {
      return std::visit(detail::overloaded {
          [](auto&& arg) -> size_t { return detail::ptr(arg)->data_offset(); },
          [](xarray_type*) -> size_t { return 0; },
          detail::throw_unsupported<std::monostate, size_t>()
      }, m_data);
    }

    Shape strides()
    {
      return std::visit(detail::overloaded {
          [](auto&& arg) -> Shape { return detail::to_Shape(detail::ptr(arg)->strides()); },
          detail::throw_unsupported<std::monostate, Shape>()
      }, m_data);
    }

    mpcf::StridedBuffer<pcf_type> strided_buffer()
    {
      mpcf::StridedBuffer<pcf_type> ret;

      ret.buffer = buffer() + offset();
      ret.shape = get_shape().data();
      ret.strides = strides().data();

      return ret;
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

    NdArray() = default;
    NdArray(xarray_type&& arr)
      : m_data(std::move(arr))
    {

    }

    static NdArray make_zeros(const Shape& shape)
    {
      NdArray arr;
      arr.m_data = xarray_type(detail::to_xshape<self_type>(shape));
      return arr;
    }

    Shape shape() const
    {
      return detail::to_Shape(m_data.shape());
    }

    Shape get_shape() const
    {
      return detail::to_Shape(m_data.shape());
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
