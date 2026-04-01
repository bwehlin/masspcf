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

#include "py_poisson.hpp"

#include <mpcf/point_process/poisson.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyPoissonBindings
  {
  public:
    using TensorT = mpcf::Tensor<mpcf::PointCloud<T>>;

    static void poisson_pp(TensorT& out, size_t dim, T rate,
                           std::vector<T> lo, std::vector<T> hi,
                           const mpcf::DefaultRandomGenerator* gen)
    {
      if (gen)
        mpcf::pp::sample_poisson(out, dim, rate, lo, hi, *gen, mpcf::default_executor());
      else
        mpcf::pp::sample_poisson(out, dim, rate, lo, hi, mpcf::default_generator(), mpcf::default_executor());
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyPoissonBindings> cls(m, ("Poisson" + suffix).c_str());

      cls
          .def_static("sample_poisson", &PyPoissonBindings::poisson_pp,
                       py::arg("out"), py::arg("dim"), py::arg("rate"),
                       py::arg("lo"), py::arg("hi"),
                       py::arg("generator").none(true) = py::none())
          ;
    }
  };

}

void mpcf_py::register_point_process_poisson(py::module_& m)
{
  PyPoissonBindings<mpcf::float32_t>::register_bindings(m, "32");
  PyPoissonBindings<mpcf::float64_t>::register_bindings(m, "64");
}
