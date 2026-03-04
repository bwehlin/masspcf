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

namespace py = pybind11;

namespace
{
  class IoOps
  {
  public:
    template <typename T>
    static void save_tensor_to_file(const mpcf::Tensor<T>& tensor, const std::string& filename)
    {
      std::ofstream file(filename, std::ios::out | std::ios::binary);
      mpcf::write(tensor, file);
    }

    template <typename T>
    static mpcf::Tensor<T> load_tensor_from_file(const std::string& filename)
    {
      std::ifstream file(filename, std::ios::in | std::ios::binary);
      return mpcf::read<mpcf::Tensor<T>>(file);
    }
  };

}

namespace mpcf_py
{

  void register_io(py::module_& m)
  {
    py::class_<IoOps>(m, "IOOps")
        .def_static("save_float32_tensor_to_file", &IoOps::save_tensor_to_file<mpcf::float32_t>)
        .def_static("load_float32_tensor_from_file", &IoOps::load_tensor_from_file<mpcf::float32_t>)

        ;
  }

}