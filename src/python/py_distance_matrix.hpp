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

#ifndef MASSPCF_PY_DISTANCE_MATRIX_H
#define MASSPCF_PY_DISTANCE_MATRIX_H

#include "pybind.hpp"
#include <pybind11/numpy.h>

#include <mpcf/distance_matrix.hpp>

#include "py_np_support.hpp"

#include <sstream>

namespace mpcf_py
{

  void register_distance_matrix(pybind11::module_& m);

  template <typename T>
  void register_distance_matrix_bindings(pybind11::module_& m, const std::string& suffix)
  {
    namespace py = pybind11;
    using MatT = mpcf::DistanceMatrix<T>;

    py::class_<MatT>(m, ("DistanceMatrix" + suffix).c_str())
      .def(py::init<size_t>(), py::arg("n"))
      .def_property_readonly("size", &MatT::size)
      .def_property_readonly("storage_count", &MatT::storage_count)
      .def("__getitem__", [](const MatT& self, std::pair<size_t, size_t> ij) {
        return self(ij.first, ij.second);
      })
      .def("__setitem__", [](MatT& self, std::pair<size_t, size_t> ij, T val) {
        self(ij.first, ij.second) = val;
      })
      .def("to_dense", [](const MatT& self) {
        auto n = self.size();
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
        oss << "DistanceMatrix" << suffix << "(size=" << self.size() << ")";
        return oss.str();
      })
    ;
  }

}

#endif // MASSPCF_PY_DISTANCE_MATRIX_H
