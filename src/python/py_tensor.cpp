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

#include "py_tensor.h"

#include <mpcf/tensor.h>
#include <mpcf/pcf.h>

#include <sstream>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

  void register_common_bindings(pybind11::module_& m)
  {
    pybind11::class_<mpcf_py::Shape>(m, "Shape")
      .def(pybind11::init<std::vector<size_t>>())
      .def(pybind11::init<>([](size_t n){ return mpcf_py::Shape{std::vector<size_t>{n}}; })) // 1d construction (Python recognizes (n) as "parenthesis int parenthesis" rather than a tuple of ints)

      .def("__eq__", [](const mpcf_py::Shape& self, pybind11::object other) {
        if (pybind11::isinstance<mpcf_py::Shape>(other))
        {
          return self.data == other.cast<mpcf_py::Shape>().data;
        }
        else if (pybind11::isinstance<std::vector<size_t>>(other))
        {
          return self.data == other.cast<std::vector<size_t>>();
        }
        else if (pybind11::isinstance<size_t>(other) || pybind11::isinstance<pybind11::int_>(other))
        {
          return self == mpcf_py::Shape(other.cast<size_t>());
        }
        else if (pybind11::isinstance<pybind11::tuple>(other))
        {
          auto t = other.cast<pybind11::tuple>();

          if (t.size() != self.data.size())
          {
            return false;
          }

          return std::equal(self.data.begin(), self.data.end(), t.begin(), [](size_t a, pybind11::handle b) {
            if (pybind11::isinstance<pybind11::int_>(b))
            {
              return a == b.cast<size_t>();
            }

            return false;
          });

        }

        std::string type_name = pybind11::str(other.get_type().attr("__name__"));
        throw std::runtime_error("Unsupported comparison with object of type " + type_name);
      })

      .def("__getitem__", &mpcf_py::Shape::dunder_getitem)
      .def("__len__", &mpcf_py::Shape::dunder_len)
      .def("__repr__", &mpcf_py::Shape::dunder_repr)
      .def("__str__", &mpcf_py::Shape::dunder_str)
    ;

    pybind11::class_<mpcf::SliceAll>(m, "SliceAll");
    pybind11::class_<mpcf::SliceIndex>(m, "SliceIndex");
    pybind11::class_<mpcf::SliceRange>(m, "SliceRange");

    m.def("slice_all", [](){ return mpcf::all(); });
    m.def("slice_index", [](ptrdiff_t idx){ return mpcf::index(idx); });
    m.def("slice_range", [](std::optional<ptrdiff_t> start, std::optional<ptrdiff_t> stop, std::optional<ptrdiff_t> step){ return mpcf::range(start, stop, step); });

  }

}

namespace mpcf_py
{
  void register_tensor_bindings(py::module_& m)
  {
    register_common_bindings(m);

    register_typed_tensor_bindings<mpcf::float64_t>(m, "Float64", "");
    register_typed_tensor_bindings<mpcf::float32_t>(m, "Float32", "");

    register_typed_tensor_bindings<mpcf::Pcf_f32>(m, "Pcf32", "");
    register_typed_tensor_bindings<mpcf::Pcf_f64>(m, "Pcf64", "");

    register_typed_tensor_bindings<mpcf::PointCloud<mpcf::float32_t>>(m, "PointCloud32", "");
    register_typed_tensor_bindings<mpcf::PointCloud<mpcf::float64_t>>(m, "PointCloud64", "");
  }
}
