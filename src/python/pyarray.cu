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
  
  template <typename Tt, typename Tv>
  class NdArray
  {
  public:
    using array_type = xt::xarray<mpcf::Pcf<Tt, Tv>>;
    using shape_type = typename array_type::shape_type;
    using pcf_type = mpcf::Pcf<Tt, Tv>;
    
    static NdArray make_zeros(const Shape& shape)
    {
      NdArray arr;
      arr.m_data = array_type(conv_shape(shape));
      return arr;
    }
    
    Shape shape() const
    {
      return conv_Shape(m_data.shape());
    }
    
  private:
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
    }
    
    array_type m_data;
  };
  
  template <typename Tt, typename Tv>
  void register_typed_array_bindings(py::handle m, const std::string& suffix)
  {
    using shape_type = typename NdArray<Tt, Tv>::shape_type;
    
    py::class_<NdArray<Tt, Tv>>(m, ("NdArray" + suffix).c_str())
        .def(py::init<>())
        .def("shape", &NdArray<Tt, Tv>::shape)
        .def_static("make_zeros", &NdArray<Tt, Tv>::make_zeros);
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
  
  py::class_<xt::xstrided_slice_vector>(m, "StridedSliceVector")
      .def(py::init<>())
      .def("append", [](xt::xstrided_slice_vector& self, size_t i){ self.push_back(i); })
      .def("append_all", [](xt::xstrided_slice_vector& self){ self.push_back(xt::all()); });
}
