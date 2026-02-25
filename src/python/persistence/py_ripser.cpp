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

#include "py_ripser.h"

#include <mpcf/tensor.h>
#include <mpcf/persistence/barcode.h>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyRipserBindings
  {
  public:
    static mpcf::Tensor<mpcf::ph::Barcode<T>> compute_barcodes_euclidean_pcloud_ripser(mpcf::Tensor<T> pointCloud)
    {
      mpcf::Tensor<mpcf::ph::Barcode<T>> ret;
      return ret;
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyRipserBindings>(m, ("PersistenceRipser" + suffix).c_str())
        .def_static("compute_barcodes_euclidean_pcloud_ripser", &PyRipserBindings::compute_barcodes_euclidean_pcloud_ripser)
      ;
    }
  };

}

namespace mpcf_py
{
  void register_persistence_ripser(pybind11::module_ &m)
  {
    PyRipserBindings<float>::register_bindings(m, "32");
    PyRipserBindings<double>::register_bindings(m, "64");
  }
}
