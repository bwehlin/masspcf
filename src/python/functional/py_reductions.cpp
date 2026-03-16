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

#include <mpcf/functional/pcf.h>
#include <mpcf/tensor.h>
#include <mpcf/algorithms/functional/matrix_reduce.h>

#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

namespace
{
  template <typename Tt, typename Tv>
  class PyReductionsBindings
  {
  public:
    using pcf_type = mpcf::Pcf<Tt, Tv>;
    using tensor_type = mpcf::Tensor<pcf_type>;

    static tensor_type mean(const tensor_type& tensor, size_t dim)
    {
      return mpcf::mean(tensor, dim);
    }

    static mpcf::Tensor<Tt> max_time(const tensor_type& tensor, size_t dim)
    {
      return mpcf::max_element(tensor, dim, [](const pcf_type& pcf){
        return pcf.points().back().t;
      }, [](Tt a, Tt b){ return std::max(a,b); });
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyReductionsBindings> cls(m, ("Reductions" + suffix).c_str());

      cls
          .def_static("mean", &PyReductionsBindings::mean)
          .def_static("max_time", &PyReductionsBindings::max_time)
          ;

    }
  };

}

namespace mpcf_py
{

  void register_reductions(py::module_& m)
  {
    PyReductionsBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
    PyReductionsBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");
  }

}
