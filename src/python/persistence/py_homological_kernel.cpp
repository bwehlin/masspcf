#include "py_homological_kernel.hpp"
#include "../py_async_support.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_homological_kernel.hpp>
#include <sbear/tensor.hpp>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyHomologicalKernelBindings
  {
  public:
    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_pcloud_task(
        const sb::Tensor<sb::PointCloud<T>> &pclouds, const sb::Tensor<sb::PointCloud<T>> &pcloudsPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<sb::PointCloud<T>, T>>(
          pclouds, pcloudsPrime, out);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_distmat_task(
        const sb::Tensor<sb::DistanceMatrix<T>> &dmats, const sb::Tensor<sb::DistanceMatrix<T>> &dmatsPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<sb::DistanceMatrix<T>, T>>(
          dmats, dmatsPrime, out);
    }

    static void register_bindings(py::module_ &m, const std::string &suffix)
    {
      py::class_<PyHomologicalKernelBindings>(m, ("HomologicalKernel" + suffix).c_str())
          .def_static(
              "spawn_homological_kernel_pcloud_task",
              &PyHomologicalKernelBindings::spawn_homological_kernel_pcloud_task)
          .def_static(
              "spawn_homological_kernel_distmat_task",
              &PyHomologicalKernelBindings::spawn_homological_kernel_distmat_task);
    }
  };

} // namespace

namespace sb_py
{
  void register_persistence_homological_kernel(pybind11::module_ &m)
  {
    PyHomologicalKernelBindings<sb::float32_t>::register_bindings(m, "32");
    PyHomologicalKernelBindings<sb::float64_t>::register_bindings(m, "64");
  }
} // namespace sb_py
