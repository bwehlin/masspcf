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

#include <mpcf/random.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "py_np_support.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace py = pybind11;

namespace
{

  template <typename Tt, typename Tv>
  class PyRandomBindings
  {
  public:
    using PcfT = mpcf::Pcf<Tt, Tv>;
    using TensorT = mpcf::Tensor<PcfT>;

    static void noisy_sin(TensorT& out, size_t nPoints)
    {
      mpcf::noisy_function(out, nPoints, [](Tv t) { return sin(2. * M_PI * t); });
    }

    static void noisy_cos(TensorT& out, size_t nPoints)
    {
      mpcf::noisy_function(out, nPoints, [](Tv t) { return cos(2. * M_PI * t); });
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyRandomBindings> cls(m, ("Random" + suffix).c_str());

      cls
          .def_static("noisy_sin", &PyRandomBindings::noisy_sin)
          .def_static("noisy_cos", &PyRandomBindings::noisy_cos)
          ;
    }
  };

}

void register_random_bindings(py::handle m)
{
  PyRandomBindings<float, float>::register_bindings(m, "_f32_f32");
  PyRandomBindings<double, double>::register_bindings(m, "_f64_f64");
}
