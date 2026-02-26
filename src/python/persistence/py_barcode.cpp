// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "py_barcode.h"
#include "../py_tensor.h"
#include "../py_np_support.h"

#include <mpcf/persistence/barcode.h>

#include <pybind11/numpy.h>

#include <sstream>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyPersistenceBarcodeBindings
  {
  public:
    using BcT = mpcf::ph::Barcode<T>;

    static BcT construct(py::array_t<T> in)
    {
      NumpyTensor<T> inTensor(in);

      if (inTensor.rank() != 2)
      {
        throw py::value_error("Input should have 2 dimensions (supplied input has " + std::to_string(inTensor.rank()) + " dimension(s)).");
      }

      if (inTensor.shape(1) != 2)
      {
        throw py::value_error("Input should have shape (n, 2). Supplied input has shape " + mpcf::shape_to_string(inTensor.shape()) + ".");
      }

      std::vector<mpcf::ph::PersistencePair<T>> bars;
      bars.reserve(inTensor.shape(0));

      for (auto i = 0_uz; i < inTensor.shape(0); ++i)
      {
        bars.emplace_back(inTensor(i, 0), inTensor(i, 1));
      }

      return BcT(std::move(bars));
    }

    [[nodiscard]] static std::string dunder_repr(const BcT& self)
    {
      std::stringstream ss;
      ss << self;
      return ss.str();
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<mpcf::ph::Barcode<T>>(m, ("Barcode" + suffix).c_str())
        .def(py::init([](py::array_t<T> arr) { return construct(arr); }))

        .def("__str__", [](const BcT& self) -> std::string{
          return "Barcode(" + PyPersistenceBarcodeBindings<T>::dunder_repr(self) + ")";
        })

        .def("__repr__", [](const BcT& self) -> std::string{
          return PyPersistenceBarcodeBindings<T>::dunder_repr(self);
        })

        .def("is_isomorphic_to", &BcT::is_isomorphic_to)
      ;
    }
  };
}

namespace mpcf_py
{

  void register_persistence_barcode_tensor(pybind11::module_ &m)
  {
    PyPersistenceBarcodeBindings<float>::register_bindings(m, "32");
    PyPersistenceBarcodeBindings<double>::register_bindings(m, "64");

    register_typed_tensor_bindings<mpcf::ph::Barcode<float>>(m, "Barcode32", "");
    register_typed_tensor_bindings<mpcf::ph::Barcode<double>>(m, "Barcode64", "");
  }
}
