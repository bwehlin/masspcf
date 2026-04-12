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

#include "pybind.hpp"
#include "py_np_support.hpp"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <mpcf/tensor.hpp>
#include <mpcf/concepts.hpp>
#include <mpcf/functional/pcf.hpp>
#include <mpcf/timeseries.hpp>
#include "functional/py_pcf_tensor_eval.hpp"

#include <algorithm>
#include <numeric>

namespace mpcf_py
{
  void register_tensor_bindings(pybind11::module_& m);

  template <typename T>
  struct scalar_of
  {
    using type = T;
  };

  template <typename T>
  requires requires { typename T::value_type; }
  struct scalar_of<T>
  {
    using type = typename T::value_type;
  };

  template <typename T>
  using scalar_of_t = typename scalar_of<T>::type;

  template <typename T>
  struct time_of
  {
    using type = T;
  };

  template <typename T>
  requires requires { typename T::time_type; }
  struct time_of<T>
  {
    using type = typename T::time_type;
  };

  template <typename T>
  using time_of_t = typename time_of<T>::type;

  class Shape
  {
  public:
    std::vector<size_t> data;

    explicit Shape(std::vector<size_t>&& shape)
      : data(std::move(shape))
    { }

    explicit Shape(const std::vector<size_t>& shape)
      : data(shape)
    { }

    explicit Shape(size_t sz) // 1d shape
      : data({sz})
    { }

    [[nodiscard]] bool operator==(const Shape& rhs) const
    {
      return data == rhs.data;
    }

    [[nodiscard]] size_t dunder_getitem(pybind11::ssize_t idx) const
    {
      auto n = static_cast<pybind11::ssize_t>(data.size());
      if (idx < 0)
        idx += n;
      if (idx < 0 || idx >= n)
        throw pybind11::index_error("Shape index out of range");
      return data[static_cast<size_t>(idx)];
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
  void assert_valid_index(const TTensor& tensor, const std::vector<size_t>& index)
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
              [](ptrdiff_t v) { return static_cast<pybind11::ssize_t>(v * sizeof(T)); });

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

      .def("__setitem__", [](TTensor& self, const std::vector<mpcf::Slice>& slices, const TTensor& vals) {
          self[slices].assign_from(vals);
        })

      .def("__eq__", [](const TTensor& self, const TTensor& rhs){
          return mpcf::elementwise_eq(self, rhs);
        })
      .def("__ne__", [](const TTensor& self, const TTensor& rhs){
          return mpcf::elementwise_ne(self, rhs);
        })
      .def("array_equal", [](const TTensor& self, const TTensor& rhs){
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
      .def("reshape", &TTensor::reshape)
      .def("transpose", &TTensor::transpose, pybind11::arg("axes") = std::vector<size_t>{})
      .def("swapaxes", &TTensor::swapaxes, pybind11::arg("axis1"), pybind11::arg("axis2"))
      .def("squeeze", [](const TTensor& self) { return self.squeeze(); })
      .def("squeeze", [](const TTensor& self, size_t axis) { return self.squeeze(axis); }, pybind11::arg("axis"))
      .def("expand_dims", &TTensor::expand_dims, pybind11::arg("axis"))
      .def_static("concatenate", [](const std::vector<TTensor>& tensors, size_t axis) {
        return mpcf::concatenate(tensors, axis);
      }, pybind11::arg("tensors"), pybind11::arg("axis") = 0)
      .def_static("stack", [](const std::vector<TTensor>& tensors, ptrdiff_t axis) {
        return mpcf::stack(tensors, axis);
      }, pybind11::arg("tensors"), pybind11::arg("axis") = 0)
      .def_static("split_sections", [](const TTensor& tensor, size_t n_sections, size_t axis) {
        return mpcf::split(tensor, n_sections, axis);
      }, pybind11::arg("tensor"), pybind11::arg("n_sections"), pybind11::arg("axis") = 0)
      .def_static("split_indices", [](const TTensor& tensor, const std::vector<size_t>& indices, size_t axis) {
        return mpcf::split(tensor, indices, axis);
      }, pybind11::arg("tensor"), pybind11::arg("indices"), pybind11::arg("axis") = 0)
      .def_static("array_split", [](const TTensor& tensor, size_t n_sections, size_t axis) {
        return mpcf::array_split(tensor, n_sections, axis);
      }, pybind11::arg("tensor"), pybind11::arg("n_sections"), pybind11::arg("axis") = 0)
      .def("is_contiguous", &TTensor::is_contiguous)
    ;

    // Unary negation
    if constexpr (mpcf::CanNegate<T>)
    {
      cls.def("__neg__", [](const TTensor& self){ return -self; });
    }

    if constexpr (mpcf::FloatType<T>)
    {
      cls.def("allclose", [](const TTensor& self, const TTensor& rhs, double atol, double rtol){
        return mpcf::allclose(self, rhs, T(atol), T(rtol));
      }, py::arg("other"), py::arg("atol") = 1e-8, py::arg("rtol") = 1e-5);
    }

    // Ordered comparisons (broadcasting, returns BoolTensor)
    if constexpr (mpcf::CanOrder<T>)
    {
      cls
        .def("__lt__", [](const TTensor& self, const TTensor& rhs){ return mpcf::elementwise_lt(self, rhs); })
        .def("__le__", [](const TTensor& self, const TTensor& rhs){ return mpcf::elementwise_le(self, rhs); })
        .def("__gt__", [](const TTensor& self, const TTensor& rhs){ return mpcf::elementwise_gt(self, rhs); })
        .def("__ge__", [](const TTensor& self, const TTensor& rhs){ return mpcf::elementwise_ge(self, rhs); })
      ;
    }

    // Tensor-Tensor arithmetic (broadcasting)
    if constexpr (mpcf::CanAddTo<T, T, T>)
    {
      cls
        .def("__add__", [](const TTensor& self, const TTensor& rhs){ return self + rhs; })
        .def("__iadd__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self += rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanSubtractTo<T, T, T>)
    {
      cls
        .def("__sub__", [](const TTensor& self, const TTensor& rhs){ return self - rhs; })
        .def("__isub__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self -= rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanMultiplyTo<T, T, T>)
    {
      cls
        .def("__mul__", [](const TTensor& self, const TTensor& rhs){ return self * rhs; })
        .def("__imul__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self *= rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanDivideTo<T, T, T>)
    {
      cls
        .def("__truediv__", [](const TTensor& self, const TTensor& rhs){ return self / rhs; })
        .def("__itruediv__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self /= rhs; return self; })
      ;
    }

    cls.def("broadcast_to", [](const TTensor& self, const std::vector<size_t>& shape){ return self.broadcast_to(shape); });

    // Masked operations
    cls.def("masked_select", [](const TTensor& self, const mpcf::Tensor<bool>& mask) {
      return mpcf::masked_select(self, mask);
    });
    cls.def("masked_assign", [](TTensor& self, const mpcf::Tensor<bool>& mask, const TTensor& values) {
      mpcf::masked_assign(self, mask, values);
    });
    cls.def("masked_fill", [](TTensor& self, const mpcf::Tensor<bool>& mask, const T& value) {
      mpcf::masked_fill(self, mask, value);
    });
    cls.def("axis_select", [](const TTensor& self, size_t axis, const mpcf::Tensor<bool>& mask) {
      return mpcf::axis_select(self, axis, mask);
    });
    cls.def("axis_assign", [](TTensor& self, size_t axis, const mpcf::Tensor<bool>& mask, const TTensor& values) {
      mpcf::axis_assign(self, axis, mask, values);
    });
    cls.def("axis_fill", [](TTensor& self, size_t axis, const mpcf::Tensor<bool>& mask, const T& value) {
      mpcf::axis_fill(self, axis, mask, value);
    });
    cls.def("multi_axis_select", [](const TTensor& self, const std::vector<std::pair<size_t, mpcf::Tensor<bool>>>& axis_masks) {
      return mpcf::multi_axis_select(self, axis_masks);
    });
    cls.def("multi_axis_assign", [](TTensor& self, const std::vector<std::pair<size_t, mpcf::Tensor<bool>>>& axis_masks, const TTensor& values) {
      mpcf::multi_axis_assign(self, axis_masks, values);
    });
    cls.def("multi_axis_fill", [](TTensor& self, const std::vector<std::pair<size_t, mpcf::Tensor<bool>>>& axis_masks, const T& value) {
      mpcf::multi_axis_fill(self, axis_masks, value);
    });

    cls.def("outer_select", [](const TTensor& self, const std::vector<std::pair<size_t, mpcf::AxisSelector>>& selectors) {
      return mpcf::outer_select(self, selectors);
    });
    cls.def("outer_assign", [](TTensor& self, const std::vector<std::pair<size_t, mpcf::AxisSelector>>& selectors, const TTensor& values) {
      mpcf::outer_assign(self, selectors, values);
    });
    cls.def("outer_fill", [](TTensor& self, const std::vector<std::pair<size_t, mpcf::AxisSelector>>& selectors, const T& value) {
      mpcf::outer_fill(self, selectors, value);
    });

    // Index-based gather/scatter (always use int64 as index type)
    cls.def("index_select", [](const TTensor& self, size_t axis, const mpcf::Tensor<mpcf::int64_t>& indices) {
      return mpcf::index_select(self, axis, indices);
    });
    cls.def("index_assign", [](TTensor& self, size_t axis, const mpcf::Tensor<mpcf::int64_t>& indices, const TTensor& values) {
      mpcf::index_assign(self, axis, indices, values);
    });
    cls.def("index_fill", [](TTensor& self, size_t axis, const mpcf::Tensor<mpcf::int64_t>& indices, const T& value) {
      mpcf::index_fill(self, axis, indices, value);
    });

    using Tv = scalar_of_t<T>;
    using Tt = time_of_t<T>;

    if constexpr (mpcf::CanAddTo<T, T, T>)
    {
      cls
        .def("__add__", [](const TTensor& self, const T& rhs){ return self + rhs; })
        .def("__radd__", [](const TTensor& self, const T& lhs){ return lhs + self; })
        .def("__iadd__", [](TTensor& self, const T& rhs) -> TTensor& { self += rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanAddTo<T, T, Tv>)
    {
      cls
        .def("__add__", [](const TTensor& self, Tv rhs){ return self + rhs; })
        .def("__iadd__", [](TTensor& self, Tv rhs) -> TTensor& { self += rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanAddTo<T, Tv, T>)
    {
      cls.def("__radd__", [](const TTensor& self, Tv lhs){ return lhs + self; });
    }

    if constexpr (mpcf::CanSubtractTo<T, T, T>)
    {
      cls
        .def("__sub__", [](const TTensor& self, const T& rhs){ return self - rhs; })
        .def("__rsub__", [](const TTensor& self, const T& lhs){ return lhs - self; })
        .def("__isub__", [](TTensor& self, const T& rhs) -> TTensor& { self -= rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanSubtractTo<T, T, Tv>)
    {
      cls
        .def("__sub__", [](const TTensor& self, Tv rhs){ return self - rhs; })
        .def("__isub__", [](TTensor& self, Tv rhs) -> TTensor& { self -= rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanSubtractTo<T, Tv, T>)
    {
      cls.def("__rsub__", [](const TTensor& self, Tv lhs){ return lhs - self; });
    }

    if constexpr (mpcf::CanMultiplyTo<T, T, Tv>)
    {
      cls
        .def("__mul__", [](const TTensor& self, Tv rhs){ return self * rhs; })
        .def("__imul__", [](TTensor& self, Tv rhs) -> TTensor& { self *= rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanMultiplyTo<T, Tv, T>)
    {
      cls.def("__rmul__", [](const TTensor& self, Tv lhs){ return lhs * self; });
    }

    if constexpr (mpcf::CanDivideTo<T, T, Tv>)
    {
      cls
        .def("__truediv__", [](const TTensor& self, Tv rhs){ return self / rhs; })
        .def("__itruediv__", [](TTensor& self, Tv rhs) -> TTensor& { self /= rhs; return self; })
      ;
    }

    if constexpr (mpcf::CanDivideTo<T, Tv, T>)
    {
      cls.def("__rtruediv__", [](const TTensor& self, Tv lhs){ return lhs / self; });
    }

    if constexpr (mpcf::CanPow<T, Tv>)
    {
      cls.def("__pow__", [](const TTensor& self, Tv exponent) {
        auto result = mpcf::pow(self, exponent);
        bool warned = result.any_of([](const T& elem) {
          if constexpr (mpcf::PcfLike<T>)
            return std::ranges::any_of(elem.points(), [](const auto& pt) {
              return std::isnan(pt.v) || std::isinf(pt.v);
            });
          else
            return std::isnan(elem) || std::isinf(elem);
        });
        if (warned)
        {
          PyErr_WarnEx(PyExc_RuntimeWarning,
            "invalid or infinite value encountered in pow", 1);
        }
        return result;
      });

      cls.def("__ipow__", [](TTensor& self, Tv exponent) -> TTensor& {
        mpcf::ipow(self, exponent);
        return self;
      });
    }

    if constexpr (mpcf::Evaluable<T, Tt, Tv>)
    {
      cls.def("__call__", [](const TTensor& self, Tt t) {
        return mpcf_py::pcf_tensor_eval_scalar<Tt, Tv>(self, t);
      });

      cls.def("__call__", [](const TTensor& self, py::array_t<Tt> times) {
        NumpyTensor<Tt> t_in(times);
        auto sh = mpcf_py::eval_out_shape(self, t_in);
        std::vector<py::ssize_t> out_shape(sh.begin(), sh.end());
        py::array_t<Tv> result(out_shape);
        NumpyTensor<Tv> out(result);
        mpcf::tensor_eval<Tt, Tv>(self, t_in, out);
        return result;
      });

      cls.def("__call__", [](const TTensor& self, const mpcf::Tensor<Tt>& times) {
        mpcf::Tensor<Tv> out(mpcf_py::eval_out_shape(self, times));
        mpcf::tensor_eval<Tt, Tv>(self, times, out);
        return out;
      });
    }

    // TimeSeries tensor evaluation (float + datetime)
    // evaluate returns Tensor<Tv>, so we use tensor_eval with Tensor<Tv> as
    // codomain and flatten the nested output to numpy.
    if constexpr (mpcf::is_timeseries_v<T>)
    {
      using TResult = mpcf::Tensor<Tv>;

      // Flatten Tensor<Tensor<Tv>> -> Tensor<Tv>.
      // tensor_shape_rank tells us where the tensor dims end and the
      // times dims begin, so channels are inserted between them:
      //   output shape = tensor_shape + (n_channels,) + times_shape
      // For 1-channel, the channel dim is squeezed.
      auto flatten_result = [](const mpcf::Tensor<TResult>& nested,
                               size_t tensor_shape_rank) {
        size_t nc = nested(std::vector<size_t>(nested.rank(), 0)).size();
        auto sh = nested.shape();

        // Build flat shape: tensor dims, then channels, then times dims
        std::vector<size_t> flat_shape;
        for (size_t i = 0; i < tensor_shape_rank; ++i)
          flat_shape.push_back(sh[i]);
        if (nc > 1)
          flat_shape.push_back(nc);
        for (size_t i = tensor_shape_rank; i < sh.size(); ++i)
          flat_shape.push_back(sh[i]);

        mpcf::Tensor<Tv> flat(flat_shape);

        mpcf::walk(nested, [&](const std::vector<size_t>& idx) {
          const auto& chan_vals = nested(idx);
          if (nc == 1) {
            flat(idx) = chan_vals(std::vector<size_t>{0});
          } else {
            // Build flat index: tensor_dims + channel + times_dims
            std::vector<size_t> flat_idx;
            for (size_t i = 0; i < tensor_shape_rank; ++i)
              flat_idx.push_back(idx[i]);
            flat_idx.push_back(0); // channel placeholder
            for (size_t i = tensor_shape_rank; i < idx.size(); ++i)
              flat_idx.push_back(idx[i]);

            for (size_t c = 0; c < nc; ++c) {
              flat_idx[tensor_shape_rank] = c;
              flat(flat_idx) = chan_vals(std::vector<size_t>{c});
            }
          }
        });
        return flat;
      };

      // Float scalar: output = tensor_shape + (n_channels,)
      cls.def("__call__", [flatten_result](const TTensor& self, Tt t) {
        auto tsr = self.rank();
        mpcf::Tensor<TResult> out(self.shape());
        mpcf::tensor_eval<Tt, TResult>(self, t, out);
        return flatten_result(out, tsr);
      });

      // Float array: output = tensor_shape + (n_channels,) + times_shape
      cls.def("__call__", [flatten_result](const TTensor& self,
                                            py::array_t<Tt> times) {
        auto tsr = self.rank();
        NumpyTensor<Tt> t_in(times);
        auto out_shape = mpcf_py::eval_out_shape(self, t_in);
        mpcf::Tensor<TResult> out(out_shape);
        mpcf::tensor_eval<Tt, TResult>(self, t_in, out);
        return flatten_result(out, tsr);
      });

      // Datetime scalar
      cls.def("__call__", [flatten_result](const TTensor& self,
                             int64_t ticks, const std::string& unit) {
        auto tsr = self.rank();
        return dispatch_datetime_unit(unit, [&](auto duration_tag) {
          using Duration = decltype(duration_tag);
          mpcf::Tensor<TResult> out(self.shape());
          mpcf::tensor_eval<Duration, TResult>(self, Duration(ticks), out);
          return flatten_result(out, tsr);
        });
      }, py::arg("ticks"), py::arg("unit"));

      // Datetime array
      cls.def("__call__", [flatten_result](const TTensor& self,
                             py::array_t<int64_t> ticks_arr,
                             const std::string& unit) {
        auto tsr = self.rank();
        return dispatch_datetime_unit(unit, [&](auto duration_tag) {
          using Duration = decltype(duration_tag);
          auto m = static_cast<size_t>(ticks_arr.size());
          mpcf::Tensor<Duration> domain(std::vector<size_t>{m});
          auto t_in = ticks_arr.template unchecked<1>();
          for (size_t i = 0; i < m; ++i)
            domain(std::vector<size_t>{i}) = Duration(t_in(i));

          auto out_shape = mpcf_py::eval_out_shape(self, domain);
          mpcf::Tensor<TResult> out(out_shape);
          mpcf::tensor_eval<Duration, TResult>(self, domain, out);
          return flatten_result(out, tsr);
        });
      }, py::arg("ticks"), py::arg("unit"));
    }

  }
}

#endif //MASSPCF_PY_TENSOR_H