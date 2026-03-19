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

#include "py_io.h"

#include <string_view>

#include <mpcf/tensor.h>
#include <mpcf/io.h>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{
  class IoOps
  {
  public:
    template <typename T>
    static void save_tensor_to_file(const mpcf::Tensor<T>& tensor, py::object file)
    {
      mpcf_py::PythonOStreamBuf buf(file);
      std::ostream os(&buf);
      mpcf::write(tensor, os);
    }

    static mpcf::io::detail::StreamableTensor load_tensor_from_file(py::object file)
    {
      mpcf_py::PythonIStreamBuf buf(file);
      std::istream is(&buf);
      return mpcf::read_any_tensor(is);
    }
  };

}

namespace mpcf_py
{

  void register_io(py::module_& m)
  {
    py::class_<IoOps>(m, "IoOps")
        .def_static("save_float32_tensor",       &IoOps::save_tensor_to_file<mpcf::float32_t>)
        .def_static("save_float64_tensor",       &IoOps::save_tensor_to_file<mpcf::float64_t>)

        .def_static("save_pcf32_tensor",         &IoOps::save_tensor_to_file<mpcf::Pcf<mpcf::float32_t, mpcf::float32_t>>)
        .def_static("save_pcf64_tensor",         &IoOps::save_tensor_to_file<mpcf::Pcf<mpcf::float64_t, mpcf::float64_t>>)

        .def_static("save_pcf32i_tensor",        &IoOps::save_tensor_to_file<mpcf::Pcf<mpcf::int32_t, mpcf::int32_t>>)
        .def_static("save_pcf64i_tensor",        &IoOps::save_tensor_to_file<mpcf::Pcf<mpcf::int64_t, mpcf::int64_t>>)

        .def_static("save_point_cloud32_tensor", &IoOps::save_tensor_to_file<mpcf::PointCloud<mpcf::float32_t>>)
        .def_static("save_point_cloud64_tensor", &IoOps::save_tensor_to_file<mpcf::PointCloud<mpcf::float64_t>>)

        .def_static("save_barcode32_tensor",     &IoOps::save_tensor_to_file<mpcf::ph::Barcode<mpcf::float32_t>>)
        .def_static("save_barcode64_tensor",     &IoOps::save_tensor_to_file<mpcf::ph::Barcode<mpcf::float64_t>>)

        .def_static("save_symmetric_matrix32_tensor", &IoOps::save_tensor_to_file<mpcf::SymmetricMatrix<mpcf::float32_t>>)
        .def_static("save_symmetric_matrix64_tensor", &IoOps::save_tensor_to_file<mpcf::SymmetricMatrix<mpcf::float64_t>>)

        .def_static("load_tensor_from_file", &IoOps::load_tensor_from_file)

        ;
  }

}