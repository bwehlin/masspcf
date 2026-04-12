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

    template <typename TOperation>
    class CrossKernelTask : public mpcf::StoppableTask<void>
    {
    public:
      CrossKernelTask(TensorT X, TensorT Y, mpcf::Tensor<Tv> out, TOperation op)
        : m_X(std::move(X)), m_Y(std::move(Y)), m_out(std::move(out)), m_op(op)
      { }

    private:
      tf::Future<void> run_async(mpcf::Executor& exec) override
      {
        auto xTotal = m_X.size();
        auto yTotal = m_Y.size();
        next_step(xTotal * yTotal, "Computing cross-kernel.", "integral");

        m_flow.for_each_index<size_t, size_t, size_t>(0ul, xTotal, 1ul, [this, yTotal](size_t xi) {
          if (stop_requested()) return;

          for (size_t yi = 0; yi < yTotal; ++yi)
          {
            m_out.flat(xi * yTotal + yi) =
                m_op(mpcf::integrate<Tt, Tv>(m_X.flat(xi), m_Y.flat(yi), m_op));
          }

          add_progress(yTotal);
        });

        return exec.cpu()->run(std::move(m_flow));
      }

      TensorT m_X;
      TensorT m_Y;
      mpcf::Tensor<Tv> m_out;
      TOperation m_op;
      tf::Taskflow m_flow;
    };

    static std::vector<PcfT> collect_all_pcfs(TensorT& tensor)
    {
      std::vector<PcfT> pcfs;
      auto total = tensor.size();
      pcfs.reserve(total);
      for (size_t i = 0; i < total; ++i)
        pcfs.push_back(tensor.flat(i));
      return pcfs;
    }

    static py::tuple l2_cross(TensorT X, TensorT Y)
    {
      auto op = mpcf::OperationL2InnerProduct<Tt, Tv>();

      std::vector<size_t> outShape;
      for (auto d : X.shape()) outShape.push_back(d);
      for (auto d : Y.shape()) outShape.push_back(d);

      auto outTensor = mpcf::Tensor<Tv>(outShape, Tv(0));

      auto xTotal = X.size();
      auto yTotal = Y.size();

      if (xTotal == 0 || yTotal == 0)
      {
        std::unique_ptr<mpcf::StoppableTask<void>> empty_task = mpcf_py::execute_empty_task();
        return py::make_tuple(std::move(empty_task), outTensor);
      }

#ifdef BUILD_WITH_CUDA
      if (!mpcf::settings().forceCpu
          && xTotal * yTotal >= mpcf::settings().cudaThreshold * mpcf::settings().cudaThreshold)
      {
        if (mpcf::settings().deviceVerbose)
        {
          std::cout << "Cross-kernel computation on CUDA device(s)" << std::endl;
        }

        auto rowPcfs = collect_all_pcfs(X);
        auto colPcfs = collect_all_pcfs(Y);
        auto task = mpcf::create_cuda_block_cdist_l2_kernel_task(outTensor, rowPcfs, colPcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->start_async(mpcf::default_executor());
        return py::make_tuple(std::move(task), outTensor);
      }
#endif

      if (mpcf::settings().deviceVerbose)
      {
        std::cout << "Cross-kernel computation on CPU(s)" << std::endl;
      }

      std::unique_ptr<mpcf::StoppableTask<void>> task =
          mpcf_py::execute_stoppable_task<CrossKernelTask<decltype(op)>>(X, Y, outTensor, op);
      return py::make_tuple(std::move(task), outTensor);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyInnerProductBindings> cls(m, ("InnerProduct" + suffix).c_str());

      cls
          .def_static("l2", &PyInnerProductBindings::l2)
          .def_static("l2_cross", &PyInnerProductBindings::l2_cross)
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
