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

#include "py_barcode_summary.h"

#include "../py_async_support.h"

#include <mpcf/functional/pcf.h>
#include <mpcf/tensor.h>
#include <mpcf/persistence/barcode.h>
#include <mpcf/persistence/stable_rank.h>
#include <mpcf/persistence/betti_curve.h>

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
          .def_static("spawn_barcode_to_stable_rank_task", [](const mpcf::Tensor<BarcodeT>& bcs, mpcf::Tensor<PcfT>& out) {
            auto task = mpcf::ph::make_stable_rank_task(bcs, out);
            task->start_async(mpcf::default_executor());
            return task;
          })
          .def_static("barcode_to_betti_curve", [](const BarcodeT& bc) {
            return mpcf::ph::barcode_to_betti_curve(bc);
          })
          .def_static("spawn_barcode_to_betti_curve_task", [](const mpcf::Tensor<BarcodeT>& bcs, mpcf::Tensor<PcfT>& out) {
            auto task = mpcf::ph::make_betti_curve_task(bcs, out);
            task->start_async(mpcf::default_executor());
            return task;
          })
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
