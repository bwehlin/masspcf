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

#include "py_distance.h"

#include "tensor.h"
#include "pcf.h"
#include "task.h"
#include "operations.cuh"
#include "algorithms/matrix_integrate.h"
#include "py_async_support.h"
#include "py_settings.h"

#include <memory>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace
{



  template <typename Tt, typename Tv>
  class PyDistanceBindings
  {
  public:
    using PcfT = mpcf::Pcf<Tt, Tv>;
    using TensorT = mpcf::Tensor<PcfT>;

    static std::unique_ptr<mpcf::StoppableTask<void>> pdist_l1(py::array_t<Tv>& matrix, TensorT fs)
    {
      auto op = mpcf::OperationL1Dist<Tt, Tv>();

      if (matrix.shape(0) == 0)
      {
        return mpcf_py::execute_empty_task();
      }

      auto* out = matrix.mutable_data(0);

#ifdef BUILD_WITH_CUDA
#if 0
      if (!mpcf_py::g_settings.forceCpu && std::distance(beginPcfs, endPcfs) >= g_settings.cudaThreshold)
      {
        if (mpcf_py::g_settings.deviceVerbose)
        {
          std::cout << "Integral computation on CUDA device(s)" << std::endl;
        }

        auto task = mpcf::create_matrix_integrate_cuda_task(out, beginPcfs, endPcfs, op, 0., std::numeric_limits<Tv>::max());
        task->set_block_dim(mpcf_py::g_settings.blockDim);
        task->start_async(mpcf::default_executor());
        return task;
      }
#endif
#endif

// TODO: deviceVerbose
#if 0
      if (g_settings.deviceVerbose)
      {
        std::cout << "Integral computation on CPU(s)" << std::endl;
      }
#endif

      auto begin = mpcf::begin1dValues(fs);
      auto end = mpcf::end1dValues(fs);

      return mpcf_py::execute_stoppable_task<mpcf::MatrixIntegrateCpuTask<decltype(op), decltype(begin)>>(out, begin, end, op);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyDistanceBindings> cls(m, ("Distance" + suffix).c_str());

      cls
          .def_static("pdist_l1", &PyDistanceBindings::pdist_l1)
      ;

    }
  };

}

namespace mpcf_py
{

  void register_distance(py::module_& m)
  {
    PyDistanceBindings<float, float>::register_bindings(m, "_f32_f32");
    PyDistanceBindings<double, double>::register_bindings(m, "_f64_f64");
  }

}