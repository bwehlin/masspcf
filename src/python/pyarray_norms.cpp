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

#include "pyarray_norms.h"
#include "algorithms/array_functional.h"
#include "algorithms/apply_functional.h"

#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

namespace
{
#if 0
  template <typename Tt, typename Tv>
  class PyBindings
  {
  public:
    using array_type = mpcf_py::NdArray<Tt, Tv>;
    using view_type = mpcf_py::View<array_type>;

    struct FuncLpNorm
    {
      template <typename T, typename OutT>
      void operator()(T expr, OutT* out, size_t outLen)
      {
        mpcf::apply_array_functional(expr, out, outLen, [](){});
      }
    };

    static void lp_norm(view_type& view, py::array_t<Tv, py::array::c_style | py::array::forcecast> out, size_t p)
    {
      auto outFlat = out.reshape({-1});
      auto outSize = outFlat.size();

      //view.apply_function<FuncLpNorm>(outFlat.data(), outSize);

    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyBindings> cls(m, ("ArrayNorms" + suffix).c_str());

      cls
          .def("lp_norm", &PyBindings::lp_norm)
      ;

    }
  };
#endif

}

namespace mpcf_py
{

  void register_array_norms(py::module_& m)
  {
    //PyBindings<float, float>::register_bindings(m, "_f32_f32");
    //PyBindings<double, double>::register_bindings(m, "_f64_f64");
  }

}