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

#include "py_ripser.hpp"
#include "../py_async_support.hpp"

#include <mpcf/tensor.hpp>
#include <mpcf/distance_matrix.hpp>
#include <mpcf/persistence/barcode.hpp>
#include <mpcf/persistence/compute_persistence.hpp>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyRipserBindings
  {
  public:
    static std::unique_ptr<mpcf::StoppableTask<void>> spawn_ripser_pcloud_euclidean_task(const mpcf::Tensor<mpcf::PointCloud<T>>& pclouds, mpcf::Tensor<mpcf::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return mpcf_py::execute_stoppable_task<mpcf::ph::RipserTask<T>>(pclouds, out, maxDim, reducedHomology);
    }

    static std::unique_ptr<mpcf::StoppableTask<void>> spawn_ripser_distmat_task(const mpcf::Tensor<mpcf::DistanceMatrix<T>>& dmats, mpcf::Tensor<mpcf::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return mpcf_py::execute_stoppable_task<mpcf::ph::RipserDistMatTask<T>>(dmats, out, maxDim, reducedHomology);
    }

#ifdef BUILD_WITH_CUDA
    static std::unique_ptr<mpcf::StoppableTask<void>> spawn_ripser_plusplus_pcloud_euclidean_task(const mpcf::Tensor<mpcf::PointCloud<T>>& pclouds, mpcf::Tensor<mpcf::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return mpcf_py::execute_stoppable_task<mpcf::ph::RipserPlusPlusTask<T>>(pclouds, out, maxDim, reducedHomology);
    }

    static std::unique_ptr<mpcf::StoppableTask<void>> spawn_ripser_plusplus_distmat_task(const mpcf::Tensor<mpcf::DistanceMatrix<T>>& dmats, mpcf::Tensor<mpcf::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return mpcf_py::execute_stoppable_task<mpcf::ph::RipserPlusPlusDistMatTask<T>>(dmats, out, maxDim, reducedHomology);
    }
#endif

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      auto cls = py::class_<PyRipserBindings>(m, ("PersistenceRipser" + suffix).c_str())
        .def_static("spawn_ripser_pcloud_euclidean_task", &PyRipserBindings::spawn_ripser_pcloud_euclidean_task)
        .def_static("spawn_ripser_distmat_task", &PyRipserBindings::spawn_ripser_distmat_task)
      ;
#ifdef BUILD_WITH_CUDA
      cls.def_static("spawn_ripser_plusplus_pcloud_euclidean_task",
                     &PyRipserBindings::spawn_ripser_plusplus_pcloud_euclidean_task);
      cls.def_static("spawn_ripser_plusplus_distmat_task",
                     &PyRipserBindings::spawn_ripser_plusplus_distmat_task);
#endif
    }
  };

}

namespace mpcf_py
{
  void register_persistence_ripser(pybind11::module_ &m)
  {
    PyRipserBindings<mpcf::float32_t>::register_bindings(m, "32");
    PyRipserBindings<mpcf::float64_t>::register_bindings(m, "64");
  }
}
