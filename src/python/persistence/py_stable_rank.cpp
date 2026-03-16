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

#include "py_stable_rank.h"

#include "../py_async_support.h"

#include <mpcf/functional/pcf.h>
#include <mpcf/tensor.h>
#include <mpcf/persistence/barcode.h>
#include <mpcf/persistence/stable_rank.h>


namespace py = pybind11;

namespace
{

  template <typename T>
  class PyStableRankBindings
  {
  public:
    using PcfT = mpcf::Pcf<T, T>;
    using BarcodeT = mpcf::ph::Barcode<T>;

    static PcfT barcode_to_stable_rank(const BarcodeT& barcode)
    {
      return mpcf::ph::barcode_to_stable_rank(barcode);
    }

    static std::unique_ptr<mpcf::StoppableTask<void>> spawn_barcode_to_stable_rank_task(const mpcf::Tensor<BarcodeT>& barcodes, mpcf::Tensor<PcfT>& out)
    {
      return mpcf_py::execute_stoppable_task<mpcf::ph::BarcodeToStableRankTask<T>>(barcodes, out);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyStableRankBindings>(m, ("PersistenceStableRank" + suffix).c_str())
          .def_static("barcode_to_stable_rank", &PyStableRankBindings::barcode_to_stable_rank)
          .def_static("spawn_barcode_to_stable_rank_task", &PyStableRankBindings::spawn_barcode_to_stable_rank_task)
          ;
    }
  };

}

namespace mpcf_py
{
  void register_persistence_stable_rank(pybind11::module_ &m)
  {
    PyStableRankBindings<mpcf::float32_t>::register_bindings(m, "32");
    PyStableRankBindings<mpcf::float64_t>::register_bindings(m, "64");
  }
}
