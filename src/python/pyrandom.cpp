/*
* Copyright 2024 Bjorn Wehlin
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
#include "pyarray.h"

#include <pybind11/pybind11.h>

#define _USE_MATH_DEFINES
#include <math.h>

namespace py = pybind11;

namespace
{
  template <typename Tt, typename Tv>
  class RandomBindings
  {
  public:
    static void noisy_sin(mpcf_py::NdArray<Tt, Tv>& out, size_t nPoints)
    {
      mpcf::noisy_function(out.data(), nPoints, [](Tv t) { return sin(2. * M_PI * t); });
    }

    static void noisy_cos(mpcf_py::NdArray<Tt, Tv>& out, size_t nPoints)
    {
      mpcf::noisy_function(out.data(), nPoints, [](Tv t) { return cos(2. * M_PI * t); });
    }
  };

  template <typename Tt, typename Tv>
  void register_types_random_bindings(py::handle m, const std::string& suffix)
  {
    using bindings = RandomBindings<Tt, Tv>;

    py::class_<bindings>(m, ("Random" + suffix).c_str())
      .def_static("noisy_sin", &bindings::noisy_sin)
      .def_static("noisy_cos", &bindings::noisy_cos);
  }
}

void register_random_bindings(py::handle m)
{
  register_types_random_bindings<float, float>(m, "_f32_f32");
  register_types_random_bindings<double, double>(m, "_f64_f64");
}
