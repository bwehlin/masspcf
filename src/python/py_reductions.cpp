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

#include "py_reductions.h"

#include <mpcf/pcf.h>
#include <mpcf/tensor.h>
#include <mpcf/algorithms/matrix_reduce.h>

#include <stdexcept>

namespace py = pybind11;

namespace
{
  template <typename Tt, typename Tv>
  class PyBindings
  {
  public:
    using pcf_type = mpcf::Pcf<Tt, Tv>;
    using tensor_type = mpcf::Tensor<pcf_type>;

    static tensor_type mean(const tensor_type& tensor, size_t dim)
    {
      return mpcf::parallel_tensor_reduce(tensor, dim);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyBindings> cls(m, ("Reductions" + suffix).c_str());

      cls
          .def_static("mean", &PyBindings::mean)
          ;

    }
  };

}

namespace mpcf_py
{

  void register_reductions(py::module_& m)
  {
    PyBindings<float, float>::register_bindings(m, "_f32_f32");
    PyBindings<double, double>::register_bindings(m, "_f64_f64");
  }

}