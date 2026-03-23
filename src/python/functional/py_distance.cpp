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
#include "functional/operations.cuh"
#include "algorithms/functional/matrix_integrate.hpp"
#include "../py_async_support.hpp"
#include "../py_settings.hpp"

#ifdef BUILD_WITH_CUDA
#include <mpcf/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <mpcf/distance_matrix.hpp>

#include <cstring>
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

    static py::tuple pdist_l1(TensorT fs)
    {
      auto op = mpcf::OperationL1Dist<Tt, Tv>();
      auto n = static_cast<size_t>(fs.shape(0));

      auto distmat = mpcf::DistanceMatrix<Tv>(n);

      if (n == 0)
      {
        std::unique_ptr<mpcf::StoppableTask<void>> empty_task = mpcf_py::execute_empty_task();
        return py::make_tuple(std::move(empty_task), distmat);
      }

      auto begin = mpcf::begin1dValues(fs);
      auto end = mpcf::end1dValues(fs);

#ifdef BUILD_WITH_CUDA
      if (!mpcf_py::g_settings.forceCpu && static_cast<size_t>(std::distance(begin, end)) >= mpcf_py::g_settings.cudaThreshold)
      {
        if (mpcf_py::g_settings.deviceVerbose)
        {
          std::cout << "Integral computation on CUDA device(s)" << std::endl;
        }

        // CUDA interim path: compute into dense numpy array, convert to DistanceMatrix on Python side
        py::array_t<Tv> dense({n, n});
        std::memset(dense.mutable_data(0), 0, n * n * sizeof(Tv));

        std::vector<PcfT> pcfs(begin, end);
        auto task = mpcf::create_cuda_matrix_integrate_l1_task(dense.mutable_data(0), pcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->set_block_dim(mpcf_py::g_settings.blockDim);
        task->start_async(mpcf::default_executor());
        return py::make_tuple(std::move(task), dense);
      }
#endif

      if (mpcf_py::g_settings.deviceVerbose)
      {
        std::cout << "Integral computation on CPU(s)" << std::endl;
      }

      std::unique_ptr<mpcf::StoppableTask<void>> task = mpcf_py::execute_stoppable_task<mpcf::MatrixIntegrateCpuDistMatTask<decltype(op), decltype(begin)>>(distmat, begin, end, op);
      return py::make_tuple(std::move(task), distmat);
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
    PyDistanceBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
    PyDistanceBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");
  }

}