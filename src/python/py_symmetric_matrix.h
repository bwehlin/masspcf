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

#pragma once

#ifndef MASSPCF_PY_SYMMETRIC_MATRIX_H
#define MASSPCF_PY_SYMMETRIC_MATRIX_H

#include "pybind.h"
#include <pybind11/numpy.h>

#include <mpcf/symmetric_matrix.h>

#include "py_np_support.h"

#include <sstream>

namespace mpcf_py
{

  void register_symmetric_matrix(pybind11::module_& m);

  template <typename T>
  void register_symmetric_matrix_bindings(pybind11::module_& m, const std::string& suffix)
  {
    namespace py = pybind11;
    using MatT = mpcf::SymmetricMatrix<T>;

    py::class_<MatT>(m, ("SymmetricMatrix" + suffix).c_str(), py::buffer_protocol())
      .def(py::init<size_t>(), py::arg("n"))
      .def(py::init<size_t, T>(), py::arg("n"), py::arg("init"))
      .def_property_readonly("n", &MatT::n)
      .def_property_readonly("storage_count", &MatT::storage_count)
      .def("__getitem__", [](const MatT& self, std::pair<size_t, size_t> ij) {
        return self(ij.first, ij.second);
      })
      .def("__setitem__", [](MatT& self, std::pair<size_t, size_t> ij, T val) {
        self(ij.first, ij.second) = val;
      })
      .def_buffer([](MatT& self) -> py::buffer_info {
        return py::buffer_info(
          self.data(),
          sizeof(T),
          py::format_descriptor<T>::format(),
          1,
          { self.storage_count() },
          { sizeof(T) }
        );
      })
      .def("to_dense", [](const MatT& self) {
        auto n = self.n();
        py::array_t<T> out({n, n});
        NumpyTensor<T> buf(out);
        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = 0; j < n; ++j)
          {
            buf(i, j) = self(i, j);
          }
        }
        return out;
      })
      .def("__repr__", [suffix](const MatT& self) {
        std::ostringstream oss;
        oss << "SymmetricMatrix" << suffix << "(n=" << self.n() << ")";
        return oss.str();
      })
    ;
  }

}

#endif // MASSPCF_PY_SYMMETRIC_MATRIX_H
