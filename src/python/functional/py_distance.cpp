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
#include <functional>
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

    using CudaFactory = std::function<std::unique_ptr<mpcf::StoppableTask<void>>(Tv*, const std::vector<PcfT>&, Tv, Tv)>;

    template <typename TOperation>
    static py::tuple pdist_impl(TensorT fs, TOperation op, CudaFactory cudaFactory)
    {
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
      if (cudaFactory && !mpcf_py::g_settings.forceCpu
          && static_cast<size_t>(std::distance(begin, end)) >= mpcf_py::g_settings.cudaThreshold)
      {
        if (mpcf_py::g_settings.deviceVerbose)
        {
          std::cout << "Integral computation on CUDA device(s)" << std::endl;
        }

        py::array_t<Tv> dense({n, n});
        std::memset(dense.mutable_data(0), 0, n * n * sizeof(Tv));

        std::vector<PcfT> pcfs(begin, end);
        auto task = cudaFactory(dense.mutable_data(0), pcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->set_block_dim(mpcf_py::g_settings.blockDim);
        task->start_async(mpcf::default_executor());
        return py::make_tuple(std::move(task), dense);
      }
#endif

      if (mpcf_py::g_settings.deviceVerbose)
      {
        std::cout << "Integral computation on CPU(s)" << std::endl;
      }

      std::unique_ptr<mpcf::StoppableTask<void>> task = mpcf_py::execute_stoppable_task<mpcf::MatrixIntegrateCpuDistMatTask<TOperation, decltype(begin)>>(distmat, begin, end, op);
      return py::make_tuple(std::move(task), distmat);
    }

    static py::tuple pdist_l1(TensorT fs)
    {
      CudaFactory cuda;
#ifdef BUILD_WITH_CUDA
      cuda = [](Tv* out, const std::vector<PcfT>& pcfs, Tv a, Tv b) {
        return mpcf::create_cuda_matrix_integrate_l1_task(out, pcfs, a, b);
      };
#endif
      return pdist_impl(fs, mpcf::OperationL1Dist<Tt, Tv>{}, cuda);
    }

    static py::tuple pdist_lp(TensorT fs, Tv p)
    {
      CudaFactory cuda;
#ifdef BUILD_WITH_CUDA
      cuda = [p](Tv* out, const std::vector<PcfT>& pcfs, Tv a, Tv b) {
        return mpcf::create_cuda_matrix_integrate_lp_task(out, pcfs, p, a, b);
      };
#endif
      return pdist_impl(fs, mpcf::OperationLpDist<Tt, Tv>(p), cuda);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyDistanceBindings> cls(m, ("Distance" + suffix).c_str());

      cls
          .def_static("pdist_l1", &PyDistanceBindings::pdist_l1)
          .def_static("pdist_lp", &PyDistanceBindings::pdist_lp)
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