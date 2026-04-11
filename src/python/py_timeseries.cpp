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

#include "py_timeseries.hpp"
#include "py_np_support.hpp"

#include <mpcf/timeseries.hpp>
#include <mpcf/algorithms/embed_time_delay.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

  /// Convert a Tensor<Tv> of shape (n,) to a numpy array.
  template <typename Tv>
  py::array_t<Tv> tensor_to_numpy(const mpcf::Tensor<Tv>& t)
  {
    auto n = t.size();
    py::array_t<Tv> result({static_cast<py::ssize_t>(n)});
    auto out = result.template mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i)
      out(i) = t(std::vector<size_t>{i});
    return result;
  }

  /// Build a TimeSeries from times (float) + values (1-D or 2-D) numpy arrays.
  /// For 1-D values: n_channels=1. For 2-D: shape (n_times, n_channels).
  /// Times are stored as offsets from times[0], time_step=1.
  template <typename Tt, typename Tv>
  mpcf::TimeSeries<Tt, Tv> build_from_times_values(
      py::array_t<Tt> times_arr, py::array_t<Tv> values_arr)
  {
    auto t = times_arr.template unchecked<1>();
    auto n_times = static_cast<size_t>(times_arr.size());
    if (n_times < 2)
      throw py::value_error("times must have at least 2 elements");

    Tt start_time = t(0);
    std::vector<Tt> times(n_times);
    for (size_t i = 0; i < n_times; ++i)
      times[i] = t(i) - start_time;

    size_t n_channels;
    std::vector<Tv> values;

    if (values_arr.ndim() == 1)
    {
      if (static_cast<size_t>(values_arr.size()) != n_times)
        throw py::value_error("times and values must have the same length");
      n_channels = 1;
      auto v = values_arr.template unchecked<1>();
      values.resize(n_times);
      for (size_t i = 0; i < n_times; ++i)
        values[i] = v(i);
    }
    else if (values_arr.ndim() == 2)
    {
      if (static_cast<size_t>(values_arr.shape(0)) != n_times)
        throw py::value_error("values rows must match times length");
      n_channels = static_cast<size_t>(values_arr.shape(1));
      auto v = values_arr.template unchecked<2>();
      values.resize(n_times * n_channels);
      for (size_t i = 0; i < n_times; ++i)
        for (size_t c = 0; c < n_channels; ++c)
          values[i * n_channels + c] = v(i, c);
    }
    else
    {
      throw py::value_error("values must be 1-D or 2-D");
    }

    return mpcf::TimeSeries<Tt, Tv>(
        std::move(times), std::move(values), n_channels, start_time, Tt(1));
  }

  /// Build from values (1-D or 2-D) + start_time + time_step (regular sampling).
  /// Times are placed at 0, 1, 2, ... with the given start_time and time_step.
  template <typename Tt, typename Tv>
  mpcf::TimeSeries<Tt, Tv> build_from_values(
      py::array_t<Tv> values_arr, Tt start_time, Tt time_step)
  {
    size_t n_times;
    size_t n_channels;
    std::vector<Tv> values;

    if (values_arr.ndim() == 1)
    {
      n_times = static_cast<size_t>(values_arr.size());
      n_channels = 1;
      auto v = values_arr.template unchecked<1>();
      values.resize(n_times);
      for (size_t i = 0; i < n_times; ++i)
        values[i] = v(i);
    }
    else if (values_arr.ndim() == 2)
    {
      n_times = static_cast<size_t>(values_arr.shape(0));
      n_channels = static_cast<size_t>(values_arr.shape(1));
      auto v = values_arr.template unchecked<2>();
      values.resize(n_times * n_channels);
      for (size_t i = 0; i < n_times; ++i)
        for (size_t c = 0; c < n_channels; ++c)
          values[i * n_channels + c] = v(i, c);
    }
    else
    {
      throw py::value_error("values must be 1-D or 2-D");
    }

    // Build times as offsets: 0, time_step, 2*time_step, ...
    // But since we store start_time separately, internal times are 0, 1, 2, ...
    std::vector<Tt> times(n_times);
    for (size_t i = 0; i < n_times; ++i)
      times[i] = static_cast<Tt>(i) * time_step;

    return mpcf::TimeSeries<Tt, Tv>(
        std::move(times), std::move(values), n_channels, start_time, Tt(1));
  }

  /// Evaluate a TimeSeries and return a numpy array.
  /// Single channel: returns scalar. Multi-channel: returns (n_channels,).
  template <typename Tt, typename Tv>
  py::object eval_scalar(const mpcf::TimeSeries<Tt, Tv>& self, Tt t)
  {
    auto result = self.evaluate(t);
    if (self.n_channels() == 1)
      return py::cast(result(std::vector<size_t>{0}));
    py::array_t<Tv> arr = tensor_to_numpy(result);
    return std::move(arr);
  }

  template <typename Tt, typename Tv>
  class PyTimeSeriesBindings
  {
  public:
    using TTimeSeries = mpcf::TimeSeries<Tt, Tv>;

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<TTimeSeries>(m, ("TimeSeries" + suffix).c_str())
        .def(py::init<>())

        // Construct from float times + values (1-D or 2-D)
        .def(py::init([](py::array_t<Tt> times, py::array_t<Tv> values) {
          return build_from_times_values<Tt, Tv>(times, values);
        }), py::arg("times"), py::arg("values"))

        // Construct from values (1-D or 2-D) + start_time + time_step
        .def(py::init([](py::array_t<Tv> values, Tt start_time, Tt time_step) {
          return build_from_values<Tt, Tv>(values, start_time, time_step);
        }), py::arg("values"), py::arg("start_time") = Tt(0),
            py::arg("time_step") = Tt(1))

        // Datetime regular sampling: values + int64 start/step + unit
        .def(py::init([](py::array_t<Tv> values,
                         int64_t start_ticks, int64_t step_ticks,
                         const std::string& unit) {
          return dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            size_t n_times, n_channels;
            std::vector<Tv> vals;
            if (values.ndim() == 1)
            {
              n_times = static_cast<size_t>(values.size());
              n_channels = 1;
              auto v = values.template unchecked<1>();
              vals.resize(n_times);
              for (size_t i = 0; i < n_times; ++i)
                vals[i] = v(i);
            }
            else
            {
              n_times = static_cast<size_t>(values.shape(0));
              n_channels = static_cast<size_t>(values.shape(1));
              auto v = values.template unchecked<2>();
              vals.resize(n_times * n_channels);
              for (size_t i = 0; i < n_times; ++i)
                for (size_t c = 0; c < n_channels; ++c)
                  vals[i * n_channels + c] = v(i, c);
            }
            std::vector<Tt> times(n_times);
            for (size_t i = 0; i < n_times; ++i)
              times[i] = static_cast<Tt>(i);
            return TTimeSeries(std::move(times), std::move(vals), n_channels,
                               Duration(start_ticks), Duration(step_ticks));
          });
        }), py::arg("values"), py::arg("start_ticks"), py::arg("step_ticks"),
            py::arg("unit"))

        // Datetime: int64 times + values + int64 step + unit
        .def(py::init([](py::array_t<int64_t> times, py::array_t<Tv> values,
                         int64_t step_ticks, const std::string& unit) {
          auto t = times.template unchecked<1>();
          auto n_times = static_cast<size_t>(times.size());
          if (n_times < 2)
            throw py::value_error("times must have at least 2 elements");

          int64_t start_ticks = t(0);
          std::vector<Tt> ts(n_times);
          for (size_t i = 0; i < n_times; ++i)
            ts[i] = static_cast<Tt>(t(i) - start_ticks)
                  / static_cast<Tt>(step_ticks);

          size_t n_channels;
          std::vector<Tv> vals;
          if (values.ndim() == 1)
          {
            if (static_cast<size_t>(values.size()) != n_times)
              throw py::value_error("times and values must have the same length");
            n_channels = 1;
            auto v = values.template unchecked<1>();
            vals.resize(n_times);
            for (size_t i = 0; i < n_times; ++i)
              vals[i] = v(i);
          }
          else
          {
            if (static_cast<size_t>(values.shape(0)) != n_times)
              throw py::value_error("values rows must match times length");
            n_channels = static_cast<size_t>(values.shape(1));
            auto v = values.template unchecked<2>();
            vals.resize(n_times * n_channels);
            for (size_t i = 0; i < n_times; ++i)
              for (size_t c = 0; c < n_channels; ++c)
                vals[i * n_channels + c] = v(i, c);
          }

          return dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            return TTimeSeries(std::move(ts), std::move(vals), n_channels,
                               Duration(start_ticks), Duration(step_ticks));
          });
        }), py::arg("times"), py::arg("values"),
            py::arg("step_ticks"), py::arg("unit"))

        // Float scalar eval
        .def("__call__", [](const TTimeSeries& self, Tt t) {
          return eval_scalar(self, t);
        })

        // Float array eval
        // Single channel: shape = times_shape
        // Multi channel: shape = (n_channels,) + times_shape
        .def("__call__", [](const TTimeSeries& self,
                            py::array_t<Tt> times) -> py::array_t<Tv> {
          auto n = static_cast<size_t>(times.size());
          auto flat = times.reshape({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tt> in(flat);
          auto nc = self.n_channels();

          // Build as (nc, n) then reshape
          py::array_t<Tv> result({static_cast<py::ssize_t>(nc),
                                  static_cast<py::ssize_t>(n)});
          auto out = result.template mutable_unchecked<2>();
          for (size_t i = 0; i < n; ++i)
          {
            auto vals = self.evaluate(in(i));
            for (size_t c = 0; c < nc; ++c)
              out(c, i) = vals(std::vector<size_t>{c});
          }

          if (nc == 1)
          {
            auto original_shape = std::vector<py::ssize_t>(
                times.shape(), times.shape() + times.ndim());
            return py::array_t<Tv>(result.reshape(original_shape));
          }

          // Shape: (n_channels,) + times_shape
          std::vector<py::ssize_t> out_shape;
          out_shape.push_back(static_cast<py::ssize_t>(nc));
          for (int d = 0; d < times.ndim(); ++d)
            out_shape.push_back(times.shape(d));
          return py::array_t<Tv>(result.reshape(out_shape));
        })

        // Datetime scalar: int64 ticks + unit string
        .def("__call__", [](const TTimeSeries& self,
                            int64_t ticks, const std::string& unit) {
          return dispatch_datetime_unit(unit, [&](auto duration_tag) -> py::object {
            using Duration = decltype(duration_tag);
            auto result = self.evaluate(Duration(ticks));
            if (self.n_channels() == 1)
              return py::cast(result(std::vector<size_t>{0}));
            py::array_t<Tv> arr = tensor_to_numpy(result);
            return std::move(arr);
          });
        }, py::arg("ticks"), py::arg("unit"))

        // Datetime array: (n_channels,) + times_shape for multi, times_shape for single
        .def("__call__", [](const TTimeSeries& self,
                            py::array_t<int64_t> ticks_arr,
                            const std::string& unit) -> py::array_t<Tv> {
          auto n = static_cast<size_t>(ticks_arr.size());
          auto flat_ticks = py::array_t<int64_t>(
              ticks_arr.reshape({static_cast<py::ssize_t>(n)}));
          auto ticks_data = flat_ticks.template unchecked<1>();
          auto nc = self.n_channels();

          py::array_t<Tv> result({static_cast<py::ssize_t>(nc),
                                  static_cast<py::ssize_t>(n)});
          auto out = result.template mutable_unchecked<2>();

          dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            for (size_t i = 0; i < n; ++i)
            {
              auto vals = self.evaluate(Duration(ticks_data(i)));
              for (size_t c = 0; c < nc; ++c)
                out(c, i) = vals(std::vector<size_t>{c});
            }
            return 0;
          });

          if (nc == 1)
          {
            auto original_shape = std::vector<py::ssize_t>(
                ticks_arr.shape(), ticks_arr.shape() + ticks_arr.ndim());
            return py::array_t<Tv>(result.reshape(original_shape));
          }
          std::vector<py::ssize_t> out_shape;
          out_shape.push_back(static_cast<py::ssize_t>(nc));
          for (int d = 0; d < ticks_arr.ndim(); ++d)
            out_shape.push_back(ticks_arr.shape(d));
          return py::array_t<Tv>(result.reshape(out_shape));
        }, py::arg("ticks"), py::arg("unit"))

        .def_property_readonly("start_time", &TTimeSeries::start_time)
        .def_property_readonly("time_step", &TTimeSeries::time_step)
        .def_property_readonly("end_time", &TTimeSeries::end_time)
        .def_property_readonly("n_channels", &TTimeSeries::n_channels)
        .def_property_readonly("n_times", &TTimeSeries::n_times)
        .def_property_readonly("_internal_times", [](const TTimeSeries& self) {
          const auto& t = self.times();
          py::array_t<Tt> result({static_cast<py::ssize_t>(t.size())});
          auto out = result.template mutable_unchecked<1>();
          for (size_t i = 0; i < t.size(); ++i)
            out(i) = t[i];
          return result;
        })

        .def("__eq__", &TTimeSeries::operator==)
        .def("__ne__", &TTimeSeries::operator!=)
        .def("__repr__", [](const TTimeSeries& self) { return self.to_string(); })
        .def("__len__", [](const TTimeSeries& self) { return self.n_times(); })
        ;
    }
  };

}

void mpcf_py::register_timeseries(pybind11::module_& m)
{
  PyTimeSeriesBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
  PyTimeSeriesBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");

  // Time delay embedding: single TimeSeries
  m.def("embed_time_delay_f32",
      [](const mpcf::TimeSeries_f32& ts, size_t dimension,
         mpcf::float32_t delay, mpcf::float32_t window,
         mpcf::float32_t stride) {
        return mpcf::embed_time_delay(ts, dimension, delay, window, stride);
      },
      py::arg("ts"), py::arg("dimension"), py::arg("delay"),
      py::arg("window") = 0.0f, py::arg("stride") = 0.0f);

  m.def("embed_time_delay_f64",
      [](const mpcf::TimeSeries_f64& ts, size_t dimension,
         mpcf::float64_t delay, mpcf::float64_t window,
         mpcf::float64_t stride) {
        return mpcf::embed_time_delay(ts, dimension, delay, window, stride);
      },
      py::arg("ts"), py::arg("dimension"), py::arg("delay"),
      py::arg("window") = 0.0, py::arg("stride") = 0.0);

  // Time delay embedding: tensor of TimeSeries
  m.def("embed_time_delay_tensor_f32",
      [](const mpcf::Tensor<mpcf::TimeSeries_f32>& ts_tensor,
         size_t dimension, mpcf::float32_t delay,
         mpcf::float32_t window, mpcf::float32_t stride) {
        return mpcf::embed_time_delay(ts_tensor, dimension, delay,
                                       window, stride);
      },
      py::arg("ts_tensor"), py::arg("dimension"), py::arg("delay"),
      py::arg("window") = 0.0f, py::arg("stride") = 0.0f);

  m.def("embed_time_delay_tensor_f64",
      [](const mpcf::Tensor<mpcf::TimeSeries_f64>& ts_tensor,
         size_t dimension, mpcf::float64_t delay,
         mpcf::float64_t window, mpcf::float64_t stride) {
        return mpcf::embed_time_delay(ts_tensor, dimension, delay,
                                       window, stride);
      },
      py::arg("ts_tensor"), py::arg("dimension"), py::arg("delay"),
      py::arg("window") = 0.0, py::arg("stride") = 0.0);
}
