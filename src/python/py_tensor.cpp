/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "py_tensor.h"

#include <mpcf/tensor.h>

#include "pyarray.h"

#include <sstream>

namespace py = pybind11;

namespace
{

  class TShape
  {
  public:
    std::vector<size_t> data;

    TShape(std::vector<size_t>&& shape)
      : data(std::move(shape))
    { }

    TShape(const std::vector<size_t>& shape)
      : data(shape)
    { }

    [[nodiscard]] size_t dunder_getitem(size_t idx) const
    {
      if (idx >= data.size())
      {
        throw py::index_error("Attempted to get index >= len");
      }
      return data[idx];
    }

    [[nodiscard]] size_t dunder_len() const noexcept
    {
      return data.size();
    }

    [[nodiscard]] std::string dunder_repr() const
    {
      std::stringstream ss;
      ss << "(";
      for (auto it = data.begin(); it != data.end(); ++it)
      {
        if (it != data.begin())
        {
          ss << ", ";
        }
        ss << *it;
      }
      ss << ")";
      return ss.str();
    }

    [[nodiscard]] std::string dunder_str() const
    {
      return "Shape" + dunder_repr();
    }
  };

  void register_common_bindings(py::handle m)
  {
    py::class_<TShape>(m, "TShape")
      .def(py::init<std::vector<size_t>>())

      .def("__getitem__", &TShape::dunder_getitem)
      .def("__len__", &TShape::dunder_len)
      .def("__repr__", &TShape::dunder_repr)
      .def("__str__", &TShape::dunder_str)
    ;
  }

  template <typename T>
  void register_typed_bindings(py::handle m, const std::string& prefix, const std::string& suffix)
  {
    using TTensor = mpcf::Tensor<T>;
    py::class_<TTensor>(m, (prefix + "Tensor" + suffix).c_str())
      .def(py::init([](const TShape& shape, const T& init)
    {
      return TTensor(shape.data, init);
    }))

      .def_property_readonly("shape", [](const TTensor& self){ return TShape{self.shape()}; })
    ;

  }

}

namespace mpcf_py
{
  void register_tensor_bindings(py::handle m)
  {
    register_common_bindings(m);

    register_typed_bindings<double>(m, "Double", "");
    register_typed_bindings<float>(m, "Float", "");
  }
}
