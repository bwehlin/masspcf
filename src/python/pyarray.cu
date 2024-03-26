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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpcf/pcf.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xlayout.hpp>

#include <vector>
#include <iostream>

namespace py = pybind11;

namespace
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
    
  private:
    std::vector<size_t> m_data;
  };
  
  namespace detail
  {
    template <typename ArrayT>
    inline typename ArrayT::shape_type to_shape(const Shape& in)
    {
      typename ArrayT::shape_type s;
      s.resize(in.size());
      std::copy(in.data().begin(), in.data().end(), s.begin());
      return s;
    }
    
    template <typename ArrayT>
    inline Shape to_Shape(const typename ArrayT::shape_type& in)
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
    using view_type = typename ArrayT::strided_view_type;
    
    StridedView(view_type data)
      : m_data(data)
    { }
    
    Shape shape() const
    {
      return detail::to_Shape<ArrayT>(m_data.shape());
    }
    
  private:
    view_type m_data;
  };
  

  
  struct StridedSliceVector
  {
    xt::xstrided_slice_vector data;
  };
  
  template <typename Tt, typename Tv>
  class NdArray
  {
  public:
    using self_type = NdArray;
    using array_type = xt::xarray<mpcf::Pcf<Tt, Tv>>;
    using shape_type = typename array_type::shape_type;
    using pcf_type = mpcf::Pcf<Tt, Tv>;
    using strided_view_type = decltype(xt::strided_view(std::declval<array_type>(), std::declval<xt::xstrided_slice_vector>()));
    
    static NdArray make_zeros(const Shape& shape)
    {
      NdArray arr;
      arr.m_data = array_type(detail::to_shape<self_type>(shape));
      return arr;
    }
    
    Shape shape() const
    {
      return detail::to_Shape<self_type>(m_data.shape());
    }
    
    strided_view_type view(const StridedSliceVector& sv)
    {
      auto view = xt::strided_view(m_data, sv.data);
      std::string asdf = view;
      return StridedView<self_type>(view);
    }
    
  private:

    /*
    static shape_type conv_shape(const Shape& in)
    {
      shape_type s;
      s.resize(in.size());
      std::copy(in.data().begin(), in.data().end(), s.begin());
      return s;
    }
    
    static Shape conv_Shape(const shape_type& in)
    {
      std::vector<size_t> s;
      s.resize(in.size());
      std::copy(in.begin(), in.end(), s.begin());
      return Shape(std::move(s));
    }*/
    
    array_type m_data;
  };
  
  template <typename Tt, typename Tv>
  void register_typed_array_bindings(py::handle m, const std::string& suffix)
  {
    using shape_type = typename NdArray<Tt, Tv>::shape_type;
    using array_type = NdArray<Tt, Tv>;
    using strided_view_type = StridedView<array_type>;
    
    py::class_<array_type>(m, ("NdArray" + suffix).c_str())
        .def(py::init<>())
        .def("shape", &array_type::shape)
        .def("view", &array_type::view)
        .def_static("make_zeros", &NdArray<Tt, Tv>::make_zeros);
    
    py::class_<strided_view_type>(m, ("StridedView" + suffix).c_str())
        .def("shape", &strided_view_type::shape);
  }

}

void register_array_bindings(py::handle m)
{
  py::class_<Shape>(m, "Shape")
      .def(py::init<>([](const std::vector<size_t>& s){ return Shape(s); }))
      .def("size", [](const Shape& self){ return self.size(); })
      .def("at", [](const Shape& self, size_t i){ return self.at(i); });
  
  register_typed_array_bindings<float, float>(m, "_f32_f32");
  register_typed_array_bindings<double, double>(m, "_f64_f64");
  
  py::class_<StridedSliceVector>(m, "StridedSliceVector")
      .def(py::init<>())
      .def("append", [](StridedSliceVector& self, size_t i){ self.data.emplace_back(i); })
      .def("append_all", [](StridedSliceVector& self){ self.data.emplace_back(xt::all()); })
      .def("append_range", [](StridedSliceVector& self, long start, long stop, long step){ 
        if (step == 1)
          self.data.emplace_back(xt::range(start, stop)); 
        else
          self.data.emplace_back(xt::range(start, stop, step));
      });
} 
