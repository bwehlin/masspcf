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

#pragma once

#ifndef MASSPCF_PY_TENSOR_H
#define MASSPCF_PY_TENSOR_H

#include "pybind.h"
#include <pybind11/stl.h>

#include <mpcf/tensor.h>

namespace mpcf_py
{
  void register_tensor_bindings(pybind11::module_& m);
  
  class Shape
  {
  public:
    std::vector<size_t> data;

    Shape(std::vector<size_t>&& shape)
      : data(std::move(shape))
    { }

    Shape(const std::vector<size_t>& shape)
      : data(shape)
    { }

    Shape(size_t sz) // 1d shape
      : data({sz})
    { }

    [[nodiscard]] bool operator==(const Shape& rhs) const
    {
      return data == rhs.data;
    }

    [[nodiscard]] size_t dunder_getitem(size_t idx) const
    {
      if (idx >= data.size())
      {
        throw pybind11::index_error("Attempted to get index >= len");
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



  template <typename TTensor>
  void assert_valid_index(const TTensor& tensor, const std::vector<size_t> index)
  {
    if (index.size() != tensor.shape().size()
      || !std::equal(index.begin(), index.end(), tensor.shape().begin(), [](size_t i, size_t s){ return i < s; })) // Check that all indices i are < shape[i]
    {
      throw pybind11::index_error("Index out of range");
    }
  }

  template <typename TTensor>
  void assert_valid_index(const TTensor& tensor, size_t index)
  {
    bool ok = tensor.shape().size() == 1 && index < tensor.shape()[0];
    if (!ok)
    {
      throw pybind11::index_error("Index out of range");
    }
  }

  template <typename T>
  void register_typed_tensor_bindings(pybind11::module_& m, const std::string& prefix, const std::string& suffix)
  {
    using TTensor = mpcf::Tensor<T>;

    pybind11::class_<TTensor> cls = [&m, &prefix, &suffix]
    {
      if constexpr (std::is_trivially_copyable_v<T>)
      {
        pybind11::class_<TTensor> cls(m, (prefix + "Tensor" + suffix).c_str(), pybind11::buffer_protocol());

        cls.def_buffer([](const TTensor& self) -> pybind11::buffer_info
        {
          if (!self.is_contiguous())
          {
            throw std::runtime_error("Noncontiguous tensor not supported.");
          }

          std::vector<pybind11::ssize_t> shape(self.shape().size(), 0);
          std::transform(self.shape().begin(), self.shape().end(), shape.begin(),
              [](size_t v) { return static_cast<pybind11::ssize_t>(v); });

          std::vector<pybind11::ssize_t> strides(self.strides().size(), 0);
          std::transform(self.strides().begin(), self.strides().end(), strides.begin(),
              [](size_t v) { return static_cast<pybind11::ssize_t>(v * sizeof(T)); });

          return pybind11::buffer_info(
              static_cast<void*>(self.data() + self.offset()),
              sizeof(T),
              pybind11::format_descriptor<T>::format(),
              self.rank(),
              shape,
              strides
          );
        });

        return cls;
      }
      else
      {
        return pybind11::class_<TTensor>(m, (prefix + "Tensor" + suffix).c_str());
      }
    }();

    cls
      .def(pybind11::init([](const Shape& shape)
        {
          return TTensor(shape.data);
        }))

      .def(pybind11::init([](const Shape& shape, const T& init)
        {
          return TTensor(shape.data, init);
        }))

      .def_property_readonly("shape", [](const TTensor& self){ return Shape{self.shape()}; })
      .def_property_readonly("strides", [](const TTensor& self){ return self.strides(); })
      .def_property_readonly("offset", [](const TTensor& self){ return self.offset(); })

      .def("__getitem__", [](const TTensor& self, const std::vector<mpcf::Slice>& slices) {
          return self[slices];
        })

      .def("__setitem__", [](const TTensor& self, const std::vector<mpcf::Slice>& slices, const TTensor& vals) {
          self[slices].assign_from(vals);
        })

      .def("__eq__", [](const TTensor& self, const TTensor& rhs){
          return self == rhs;
        })

      .def("_get_element", [](const TTensor& self, const std::vector<size_t>& index) {
          assert_valid_index(self, index);
          return self(index);
        })

      .def("_get_element", [](TTensor& self, size_t index) {
          assert_valid_index(self, index);
          return self(index);
        })

      .def("_set_element", [](TTensor& self, const std::vector<size_t>& index, const T& val) {
          assert_valid_index(self, index);
          self(index) = val;
        })

      .def("copy", &TTensor::copy)
      .def("flatten", &TTensor::flatten)
    ;

  }
}

#endif //MASSPCF_PY_TENSOR_H