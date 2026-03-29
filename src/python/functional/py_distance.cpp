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

#include <taskflow/algorithm/for_each.hpp>
#include "../py_async_support.hpp"
#include "../py_settings.hpp"

#ifdef BUILD_WITH_CUDA
#include <mpcf/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <mpcf/distance_matrix.hpp>

#include <functional>
#include <memory>

namespace py = pybind11;

namespace
{



  template <typename Tt, typename Tv>
  class PyDistanceBindings
  {
  public:
    using PcfT = mpcf::Pcf<Tt, Tv>;
    using TensorT = mpcf::Tensor<PcfT>;

    using CudaFactory = std::function<std::unique_ptr<mpcf::StoppableTask<void>>(mpcf::DistanceMatrix<Tv>&, const std::vector<PcfT>&, Tv, Tv)>;

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

        std::vector<PcfT> pcfs(begin, end);
        auto task = cudaFactory(distmat, pcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->set_block_dim(mpcf_py::g_settings.blockDim);
        task->start_async(mpcf::default_executor());
        return py::make_tuple(std::move(task), distmat);
      }
#endif

      if (mpcf_py::g_settings.deviceVerbose)
      {
        std::cout << "Integral computation on CPU(s)" << std::endl;
      }

      std::unique_ptr<mpcf::StoppableTask<void>> task = mpcf_py::execute_stoppable_task<mpcf::CpuPairwiseIntegrationTask<TOperation, decltype(begin), mpcf::DistanceMatrix<Tv>, false>>(distmat, begin, end, op);
      return py::make_tuple(std::move(task), distmat);
    }

    static py::tuple pdist_l1(TensorT fs)
    {
      CudaFactory cuda;
#ifdef BUILD_WITH_CUDA
      cuda = [](mpcf::DistanceMatrix<Tv>& out, const std::vector<PcfT>& pcfs, Tv a, Tv b) {
        return mpcf::create_cuda_block_integrate_l1_task(out, pcfs, a, b);
      };
#endif
      return pdist_impl(fs, mpcf::OperationL1Dist<Tt, Tv>{}, cuda);
    }

    static py::tuple pdist_lp(TensorT fs, Tv p)
    {
      CudaFactory cuda;
#ifdef BUILD_WITH_CUDA
      cuda = [p](mpcf::DistanceMatrix<Tv>& out, const std::vector<PcfT>& pcfs, Tv a, Tv b) {
        return mpcf::create_cuda_block_integrate_lp_task(out, pcfs, p, a, b);
      };
#endif
      return pdist_impl(fs, mpcf::OperationLpDist<Tt, Tv>(p), cuda);
    }

    template <typename TOperation>
    class CdistTask : public mpcf::StoppableTask<void>
    {
    public:
      CdistTask(TensorT X, TensorT Y, mpcf::Tensor<Tv> out, TOperation op)
        : m_X(std::move(X)), m_Y(std::move(Y)), m_out(std::move(out)), m_op(op)
      { }

    private:
      tf::Future<void> run_async(mpcf::Executor& exec) override
      {
        auto xTotal = m_X.size();
        auto yTotal = m_Y.size();
        next_step(xTotal * yTotal, "Computing cross-distances.", "integral");

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

    using CudaCdistFactory = std::function<std::unique_ptr<mpcf::StoppableTask<void>>(
        mpcf::Tensor<Tv>&, const std::vector<PcfT>&, const std::vector<PcfT>&, Tv, Tv)>;

    static std::vector<PcfT> collect_all_pcfs(TensorT& tensor)
    {
      std::vector<PcfT> pcfs;
      auto total = tensor.size();
      pcfs.reserve(total);
      for (size_t i = 0; i < total; ++i)
      {
        pcfs.push_back(tensor.flat(i));
      }
      return pcfs;
    }

    template <typename TOperation>
    static py::tuple cdist_impl(TensorT X, TensorT Y, TOperation op, CudaCdistFactory cudaFactory)
    {
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
      if (cudaFactory && !mpcf_py::g_settings.forceCpu
          && xTotal * yTotal >= mpcf_py::g_settings.cudaThreshold * mpcf_py::g_settings.cudaThreshold)
      {
        if (mpcf_py::g_settings.deviceVerbose)
        {
          std::cout << "Cross-distance computation on CUDA device(s)" << std::endl;
        }

        auto rowPcfs = collect_all_pcfs(X);
        auto colPcfs = collect_all_pcfs(Y);
        auto task = cudaFactory(outTensor, rowPcfs, colPcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->set_block_dim(mpcf_py::g_settings.blockDim);
        task->start_async(mpcf::default_executor());
        return py::make_tuple(std::move(task), outTensor);
      }
#endif

      std::unique_ptr<mpcf::StoppableTask<void>> task =
          mpcf_py::execute_stoppable_task<CdistTask<TOperation>>(X, Y, outTensor, op);
      return py::make_tuple(std::move(task), outTensor);
    }

    static Tv lp_distance_l1(const PcfT& f, const PcfT& g)
    {
      mpcf::OperationL1Dist<Tt, Tv> op;
      return op(mpcf::integrate<Tt, Tv>(f, g, op));
    }

    static Tv lp_distance_lp(const PcfT& f, const PcfT& g, Tv p)
    {
      mpcf::OperationLpDist<Tt, Tv> op(p);
      return op(mpcf::integrate<Tt, Tv>(f, g, op));
    }

    static py::tuple cdist_l1(TensorT X, TensorT Y)
    {
      CudaCdistFactory cuda;
#ifdef BUILD_WITH_CUDA
      cuda = [](mpcf::Tensor<Tv>& out, const std::vector<PcfT>& rows, const std::vector<PcfT>& cols, Tv a, Tv b) {
        return mpcf::create_cuda_block_cdist_l1_task(out, rows, cols, a, b);
      };
#endif
      return cdist_impl(X, Y, mpcf::OperationL1Dist<Tt, Tv>{}, cuda);
    }

    static py::tuple cdist_lp(TensorT X, TensorT Y, Tv p)
    {
      CudaCdistFactory cuda;
#ifdef BUILD_WITH_CUDA
      cuda = [p](mpcf::Tensor<Tv>& out, const std::vector<PcfT>& rows, const std::vector<PcfT>& cols, Tv a, Tv b) {
        return mpcf::create_cuda_block_cdist_lp_task(out, rows, cols, p, a, b);
      };
#endif
      return cdist_impl(X, Y, mpcf::OperationLpDist<Tt, Tv>(p), cuda);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyDistanceBindings> cls(m, ("Distance" + suffix).c_str());

      cls
          .def_static("pdist_l1", &PyDistanceBindings::pdist_l1)
          .def_static("pdist_lp", &PyDistanceBindings::pdist_lp)
          .def_static("lp_distance_l1", &PyDistanceBindings::lp_distance_l1)
          .def_static("lp_distance_lp", &PyDistanceBindings::lp_distance_lp)
          .def_static("cdist_l1", &PyDistanceBindings::cdist_l1)
          .def_static("cdist_lp", &PyDistanceBindings::cdist_lp)
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