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

#include "pyarray.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpcf/pcf.h>

#include <vector>
#include <iostream>

namespace py = pybind11;

namespace
{
  template <typename Tt, typename Tv>
  void register_typed_array_bindings(py::handle m, const std::string& suffix)
  {
    using xshape_type = typename mpcf_py::NdArray<Tt, Tv>::xshape_type;
    using array_type = mpcf_py::NdArray<Tt, Tv>;
    using view_type = mpcf_py::View<array_type>;

    py::class_<array_type>(m, ("NdArray" + suffix).c_str())
        .def(py::init<>())
        .def("as_view", &array_type::as_view, py::keep_alive<0, 1>())
        .def("shape", &array_type::shape)
        .def("strided_view", &array_type::strided_view, py::keep_alive<0, 1>()) // keep NdArray alive for at least as long as View
        .def("at", &array_type::at, py::keep_alive<0, 1>())
        .def_static("make_zeros", &mpcf_py::NdArray<Tt, Tv>::make_zeros);
    
    py::class_<view_type>(m, ("View" + suffix).c_str())
      .def("strided_view", &view_type::strided_view, py::keep_alive<0, 1>())
      .def("shape", &view_type::get_shape)
      .def("transpose", &view_type::transpose)
      .def("assign", &view_type::assign)
      .def("at", &view_type::at)

      .def("reduce_mean", &view_type::reduce_mean);
  }

}

void register_array_bindings(py::handle m)
{
  py::class_<mpcf_py::Shape>(m, "Shape")
      .def(py::init<>([](const std::vector<size_t>& s){ return mpcf_py::Shape(s); }))
      .def("size", [](const mpcf_py::Shape& self){ return self.size(); })
      .def("at", [](const mpcf_py::Shape& self, size_t i){ return self.at(i); });
  
  register_typed_array_bindings<float, float>(m, "_f32_f32");
  register_typed_array_bindings<double, double>(m, "_f64_f64");
  
  py::class_<mpcf_py::StridedSliceVector>(m, "StridedSliceVector")
      .def(py::init<>())
      .def("append", [](mpcf_py::StridedSliceVector& self, size_t i){ self.data.emplace_back(i); })
      .def("append_all", &mpcf_py::StridedSliceVector::append_all)
      .def("append_range", &mpcf_py::StridedSliceVector::append_range)
      .def("append_range_from", &mpcf_py::StridedSliceVector::append_range_from)
      .def("append_range_to", &mpcf_py::StridedSliceVector::append_range_to)
      ;
}
