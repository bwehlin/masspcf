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

#include "py_barcode_summary.hpp"

#include "../py_async_support.hpp"

#include <mpcf/functional/pcf.hpp>
#include <mpcf/tensor.hpp>
#include <mpcf/persistence/barcode.hpp>
#include <mpcf/persistence/accumulated_persistence.hpp>
#include <mpcf/persistence/stable_rank.hpp>
#include <mpcf/persistence/betti_curve.hpp>
#include <mpcf/persistence/filter_significant.hpp>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyBarcodeSummaryBindings
  {
  public:
    using PcfT = mpcf::Pcf<T, T>;
    using BarcodeT = mpcf::ph::Barcode<T>;

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyBarcodeSummaryBindings>(m, ("PersistenceBarcodeSummary" + suffix).c_str())
          .def_static("barcode_to_stable_rank", [](const BarcodeT& bc) {
            return mpcf::ph::barcode_to_stable_rank(bc);
          })
          .def_static("spawn_barcode_to_stable_rank_task", [](const mpcf::Tensor<BarcodeT>& bcs, mpcf::Tensor<PcfT>& out)
              -> std::unique_ptr<mpcf::StoppableTask<void>> {
            auto task = mpcf::ph::make_stable_rank_task(bcs, out);
            task->start_async(mpcf::default_executor());
            return task;
          })
          .def_static("barcode_to_betti_curve", [](const BarcodeT& bc) {
            return mpcf::ph::barcode_to_betti_curve(bc);
          })
          .def_static("spawn_barcode_to_betti_curve_task", [](const mpcf::Tensor<BarcodeT>& bcs, mpcf::Tensor<PcfT>& out)
              -> std::unique_ptr<mpcf::StoppableTask<void>> {
            auto task = mpcf::ph::make_betti_curve_task(bcs, out);
            task->start_async(mpcf::default_executor());
            return task;
          })
          .def_static("barcode_to_accumulated_persistence", [](const BarcodeT& bc, T max_death) {
            return mpcf::ph::barcode_to_accumulated_persistence(bc, max_death);
          }, py::arg("barcode"), py::arg("max_death") = std::numeric_limits<T>::infinity())
          .def_static("spawn_barcode_to_accumulated_persistence_task", [](const mpcf::Tensor<BarcodeT>& bcs, mpcf::Tensor<PcfT>& out, T max_death)
              -> std::unique_ptr<mpcf::StoppableTask<void>> {
            auto task = mpcf::ph::make_accumulated_persistence_task(bcs, out, max_death);
            task->start_async(mpcf::default_executor());
            return task;
          }, py::arg("barcodes"), py::arg("out"), py::arg("max_death") = std::numeric_limits<T>::infinity())
          .def_static("filter_significant_bars", [](const BarcodeT& bc, T alpha) {
            return mpcf::ph::filter_significant_bars(bc, alpha);
          }, py::arg("barcode"), py::arg("alpha") = T(0.05))
          .def_static("spawn_filter_significant_task", [](const mpcf::Tensor<BarcodeT>& bcs, mpcf::Tensor<BarcodeT>& out, T alpha)
              -> std::unique_ptr<mpcf::StoppableTask<void>> {
            auto task = mpcf::ph::make_filter_significant_task(bcs, out, alpha);
            task->start_async(mpcf::default_executor());
            return task;
          }, py::arg("barcodes"), py::arg("out"), py::arg("alpha") = T(0.05))
          ;
    }
  };

}

namespace mpcf_py
{
  void register_persistence_barcode_summary(pybind11::module_ &m)
  {
    PyBarcodeSummaryBindings<mpcf::float32_t>::register_bindings(m, "32");
    PyBarcodeSummaryBindings<mpcf::float64_t>::register_bindings(m, "64");
  }
}
