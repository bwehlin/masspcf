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

#include "py_distance.hpp"

#include "tensor.hpp"
#include "functional/pcf.hpp"
#include "task.hpp"
#include "../py_async_support.hpp"
#include "../py_np_support.hpp"
#include "algorithms/functional/apply_functional.hpp"

#include <memory>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace
{



  template <typename Tt, typename Tv>
  class PyNormsBindings
  {
  public:
    using PcfT = mpcf::Pcf<Tt, Tv>;
    using TensorT = mpcf::Tensor<PcfT>;

    static std::unique_ptr<mpcf::StoppableTask<void>> lpnorm_l1(py::array_t<Tv>& out, TensorT inTensor)
    {
      NumpyTensor<Tv> outTensor(out);

      auto f = [](const PcfT& v) {
        return mpcf::l1_norm(v);
      };

      return mpcf_py::execute_stoppable_task<mpcf::ApplyFunctional<TensorT, NumpyTensor<Tv>, decltype(f)>>(inTensor, outTensor, f);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyNormsBindings> cls(m, ("Norms" + suffix).c_str());

      cls
          .def_static("lpnorm_l1", &PyNormsBindings::lpnorm_l1)
      ;

    }
  };

}

namespace mpcf_py
{

  void register_norms(py::module_& m)
  {
    PyNormsBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
    PyNormsBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");
  }

}