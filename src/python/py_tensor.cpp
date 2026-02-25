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
#include <mpcf/pcf.h>

#include <sstream>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

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

  void register_common_bindings(py::module_& m)
  {
    py::class_<Shape>(m, "Shape")
      .def(py::init<std::vector<size_t>>())
      .def(py::init<>([](size_t n){ return Shape{std::vector<size_t>{n}}; })) // 1d construction (Python recognizes (n) as "parenthesis int parenthesis" rather than a tuple of ints)

      .def("__eq__", [](const Shape& self, py::object other) {
        if (py::isinstance<Shape>(other))
        {
          return self.data == other.cast<Shape>().data;
        }
        else if (py::isinstance<std::vector<size_t>>(other))
        {
          return self.data == other.cast<std::vector<size_t>>();
        }
        else if (py::isinstance<size_t>(other) || py::isinstance<py::int_>(other))
        {
          return self == Shape(other.cast<size_t>());
        }
        else if (py::isinstance<py::tuple>(other))
        {
          auto t = other.cast<py::tuple>();

          if (t.size() != self.data.size())
          {
            return false;
          }

          return std::equal(self.data.begin(), self.data.end(), t.begin(), [](size_t a, py::handle b) {
            if (py::isinstance<py::int_>(b))
            {
              return a == b.cast<size_t>();
            }

            return false;
          });

        }

        std::string type_name = py::str(other.get_type().attr("__name__"));
        throw std::runtime_error("Unsupported comparison with object of type " + type_name);
      })

      .def("__getitem__", &Shape::dunder_getitem)
      .def("__len__", &Shape::dunder_len)
      .def("__repr__", &Shape::dunder_repr)
      .def("__str__", &Shape::dunder_str)
    ;

    py::class_<mpcf::SliceAll>(m, "SliceAll");
    py::class_<mpcf::SliceIndex>(m, "SliceIndex");
    py::class_<mpcf::SliceRange>(m, "SliceRange");

    m.def("slice_all", [](){ return mpcf::all(); });
    m.def("slice_index", [](ptrdiff_t idx){ return mpcf::index(idx); });
    m.def("slice_range", [](std::optional<ptrdiff_t> start, std::optional<ptrdiff_t> stop, std::optional<ptrdiff_t> step){ return mpcf::range(start, stop, step); });

  }

  template <typename TTensor>
  void assert_valid_index(const TTensor& tensor, const std::vector<size_t> index)
  {
    if (index.size() != tensor.shape().size()
      || !std::equal(index.begin(), index.end(), tensor.shape().begin(), [](size_t i, size_t s){ return i < s; })) // Check that all indices i are < shape[i]
    {
      throw py::index_error("Index out of range");
    }
  }

  template <typename TTensor>
  void assert_valid_index(const TTensor& tensor, size_t index)
  {
    bool ok = tensor.shape().size() == 1 && index < tensor.shape()[0];
    if (!ok)
    {
      throw py::index_error("Index out of range");
    }
  }

  template <typename T>
  void register_typed_bindings(py::module_& m, const std::string& prefix, const std::string& suffix)
  {
    using TTensor = mpcf::Tensor<T>;

    py::class_<TTensor> cls = [&m, &prefix, &suffix]
    {
      if constexpr (std::is_trivially_copyable_v<T>)
      {
        py::class_<TTensor> cls(m, (prefix + "Tensor" + suffix).c_str(), py::buffer_protocol());

        cls.def_buffer([](const TTensor& self) -> py::buffer_info
        {
          if (!self.is_contiguous())
          {
            throw std::runtime_error("Noncontiguous tensor not supported.");
          }

          std::vector<py::ssize_t> shape(self.shape().size(), 0);
          std::transform(self.shape().begin(), self.shape().end(), shape.begin(),
              [](size_t v) { return static_cast<py::ssize_t>(v); });

          std::vector<py::ssize_t> strides(self.strides().size(), 0);
          std::transform(self.strides().begin(), self.strides().end(), strides.begin(),
              [](size_t v) { return static_cast<py::ssize_t>(v); });

          return py::buffer_info(
              static_cast<void*>(self.data() + self.offset()),
              sizeof(T),
              py::format_descriptor<T>::format(),
              self.rank(),
              self.shape(),
              self.strides()
          );
        });

        return cls;
      }
      else
      {
        return py::class_<TTensor>(m, (prefix + "Tensor" + suffix).c_str());
      }
    }();

    cls
      .def(py::init([](const Shape& shape)
        {
          return TTensor(shape.data);
        }))

      .def(py::init([](const Shape& shape, const T& init)
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

namespace mpcf_py
{
  void register_tensor_bindings(py::module_& m)
  {
    register_common_bindings(m);

    register_typed_bindings<double>(m, "Double", "");
    register_typed_bindings<float>(m, "Float", "");

    register_typed_bindings<mpcf::Pcf_f32>(m, "Pcf32", "");
    register_typed_bindings<mpcf::Pcf_f64>(m, "Pcf64", "");

    register_typed_bindings<mpcf::Tensor<float>>(m, "PointCloud32", "");
    register_typed_bindings<mpcf::Tensor<double>>(m, "PointCloud64", "");
  }
}
