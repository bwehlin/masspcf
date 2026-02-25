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

#include "py_barcode_tensor.h"
#include "../py_tensor.h"

#include <mpcf/persistence/barcode.h>
#include <mpcf/persistence/persistence_pair.h>

namespace py = pybind11;

namespace
{
  template <typename T>
  class PyPersistenceBarcodeBindings
  {
  public:
    static void register_bindings(py::module_& m, const std::string& name)
    {

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
