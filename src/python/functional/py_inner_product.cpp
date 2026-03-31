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

#include "py_inner_product.hpp"

#include "tensor.hpp"
#include "functional/pcf.hpp"
#include "task.hpp"
#include "functional/operations.cuh"
#include "algorithms/functional/matrix_integrate.hpp"
#include "../py_async_support.hpp"
#include <mpcf/settings.hpp>

#ifdef BUILD_WITH_CUDA
#include <mpcf/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <mpcf/symmetric_matrix.hpp>

#include <memory>

namespace py = pybind11;

namespace
{

  template <typename Tt, typename Tv>
  class PyInnerProductBindings
  {
  public:
    using PcfT = mpcf::Pcf<Tt, Tv>;
    using TensorT = mpcf::Tensor<PcfT>;

    static py::tuple l2(TensorT fs)
    {
      auto op = mpcf::OperationL2InnerProduct<Tt, Tv>();
      auto n = static_cast<size_t>(fs.shape(0));

      auto symmat = mpcf::SymmetricMatrix<Tv>(n);

      if (n == 0)
      {
        std::unique_ptr<mpcf::StoppableTask<void>> empty_task = mpcf_py::execute_empty_task();
        return py::make_tuple(std::move(empty_task), symmat);
      }

      auto begin = mpcf::begin1dValues(fs);
      auto end = mpcf::end1dValues(fs);

#ifdef BUILD_WITH_CUDA
      if (!mpcf::settings().forceCpu && static_cast<size_t>(std::distance(begin, end)) >= mpcf::settings().cudaThreshold)
      {
        if (mpcf::settings().deviceVerbose)
        {
          std::cout << "Kernel computation on CUDA device(s)" << std::endl;
        }

        std::vector<PcfT> pcfs(begin, end);
        auto task = mpcf::create_cuda_block_integrate_l2_kernel_task(symmat, pcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->start_async(mpcf::default_executor());
        return py::make_tuple(std::move(task), symmat);
      }
#endif

      if (mpcf::settings().deviceVerbose)
      {
        std::cout << "Kernel computation on CPU(s)" << std::endl;
      }

      std::unique_ptr<mpcf::StoppableTask<void>> task = mpcf_py::execute_stoppable_task<mpcf::CpuPairwiseIntegrationTask<decltype(op), decltype(begin), mpcf::SymmetricMatrix<Tv>, true>>(symmat, begin, end, op);
      return py::make_tuple(std::move(task), symmat);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyInnerProductBindings> cls(m, ("InnerProduct" + suffix).c_str());

      cls
          .def_static("l2", &PyInnerProductBindings::l2)
      ;
    }
  };

}

namespace mpcf_py
{

  void register_inner_product(py::module_& m)
  {
    PyInnerProductBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
    PyInnerProductBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");
  }

}
