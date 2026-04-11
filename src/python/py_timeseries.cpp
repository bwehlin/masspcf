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
#include "pypcf_support.hpp"

#include <mpcf/timeseries.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

  template <typename Tt, typename Tv>
  class PyTimeSeriesBindings
  {
  public:
    using TTimeSeries = mpcf::TimeSeries<Tt, Tv>;
    using TPcf = mpcf::Pcf<Tt, Tv>;

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<TTimeSeries>(m, ("TimeSeries" + suffix).c_str())
        .def(py::init<>())

        .def(py::init([](const TPcf& pcf, Tt start_time, Tt time_step) {
          return TTimeSeries(TPcf(pcf), start_time, time_step);
        }), py::arg("pcf"), py::arg("start_time") = Tt(0), py::arg("time_step") = Tt(1))

        // Construct from 1-D values + start_time + time_step (regular sampling)
        .def(py::init([](py::array_t<Tv> values, Tt start_time, Tt time_step) {
          auto v = values.template unchecked<1>();
          std::vector<mpcf::TimePoint<Tt, Tv>> points;
          points.reserve(values.size());
          for (py::ssize_t i = 0; i < values.size(); ++i)
          {
            points.emplace_back(static_cast<Tt>(i), v(i));
          }
          return TTimeSeries(TPcf(std::move(points)), start_time, time_step);
        }), py::arg("values"), py::arg("start_time") = Tt(0), py::arg("time_step") = Tt(1))

        // Datetime regular sampling: 1-D values + int64 start/step + unit
        .def(py::init([](py::array_t<Tv> values,
                         int64_t start_ticks, int64_t step_ticks,
                         const std::string& unit) {
          auto v = values.template unchecked<1>();
          std::vector<mpcf::TimePoint<Tt, Tv>> points;
          points.reserve(values.size());
          for (py::ssize_t i = 0; i < values.size(); ++i)
          {
            points.emplace_back(static_cast<Tt>(i), v(i));
          }
          return dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            return TTimeSeries(TPcf(std::move(points)),
                               Duration(start_ticks), Duration(step_ticks));
          });
        }), py::arg("values"), py::arg("start_ticks"), py::arg("step_ticks"),
            py::arg("unit"))

        // Construct from separate times and values arrays.
        // PCF breakpoints store real offsets from start, time_step=1.
        .def(py::init([](py::array_t<Tt> times, py::array_t<Tv> values) {
          auto t = times.template unchecked<1>();
          auto v = values.template unchecked<1>();
          if (times.size() != values.size())
            throw py::value_error("times and values must have the same length");
          if (times.size() < 2)
            throw py::value_error("times must have at least 2 elements");

          Tt start_time = t(0);
          std::vector<mpcf::TimePoint<Tt, Tv>> points;
          points.reserve(times.size());
          for (py::ssize_t i = 0; i < times.size(); ++i)
          {
            points.emplace_back(t(i) - start_time, v(i));
          }
          return TTimeSeries(TPcf(std::move(points)), start_time, Tt(1));
        }), py::arg("times"), py::arg("values"))

        // Datetime: int64 times + float values + int64 step + unit
        .def(py::init([](py::array_t<int64_t> times, py::array_t<Tv> values,
                         int64_t step_ticks, const std::string& unit) {
          auto t = times.template unchecked<1>();
          auto v = values.template unchecked<1>();
          if (times.size() != values.size())
            throw py::value_error("times and values must have the same length");
          if (times.size() < 2)
            throw py::value_error("times must have at least 2 elements");

          int64_t start_ticks = t(0);
          std::vector<mpcf::TimePoint<Tt, Tv>> points;
          points.reserve(times.size());
          for (py::ssize_t i = 0; i < times.size(); ++i)
          {
            Tt pcf_t = static_cast<Tt>(t(i) - start_ticks)
                     / static_cast<Tt>(step_ticks);
            points.emplace_back(pcf_t, v(i));
          }
          return dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            return TTimeSeries(TPcf(std::move(points)),
                               Duration(start_ticks), Duration(step_ticks));
          });
        }), py::arg("times"), py::arg("values"),
            py::arg("step_ticks"), py::arg("unit"))

        .def("__call__", [](const TTimeSeries& self, Tt t) -> Tv {
          return self.evaluate(t);
        })

        // Datetime scalar: int64 ticks + unit string
        .def("__call__", [](const TTimeSeries& self,
                            int64_t ticks, const std::string& unit) -> Tv {
          return dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            return self.evaluate(Duration(ticks));
          });
        }, py::arg("ticks"), py::arg("unit"))

        // Datetime array: int64 numpy array + unit string
        .def("__call__", [](const TTimeSeries& self,
                            py::array_t<int64_t> ticks_arr,
                            const std::string& unit) -> py::array_t<Tv> {
          auto original_shape = std::vector<py::ssize_t>(
              ticks_arr.shape(), ticks_arr.shape() + ticks_arr.ndim());
          auto n = static_cast<size_t>(ticks_arr.size());
          auto flat_ticks = py::array_t<int64_t>(
              ticks_arr.reshape({static_cast<py::ssize_t>(n)}));
          auto ticks_data = flat_ticks.template unchecked<1>();

          py::array_t<Tv> flat_result({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tv> out(flat_result);

          dispatch_datetime_unit(unit, [&](auto duration_tag) {
            using Duration = decltype(duration_tag);
            for (size_t i = 0; i < n; ++i)
            {
              out(i) = self.evaluate(Duration(ticks_data(i)));
            }
            return Tv(0); // dummy return for dispatch
          });

          return flat_result.reshape(original_shape);
        }, py::arg("ticks"), py::arg("unit"))

        .def("__call__", [](const TTimeSeries& self, py::array_t<Tt> times) -> py::array_t<Tv> {
          auto original_shape = std::vector<py::ssize_t>(times.shape(), times.shape() + times.ndim());
          auto n = static_cast<size_t>(times.size());
          auto flat_times = times.reshape({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tt> in(flat_times);

          py::array_t<Tv> flat_result({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tv> out(flat_result);
          for (size_t i = 0; i < n; ++i)
          {
            out(i) = self.evaluate(in(i));
          }

          return flat_result.reshape(original_shape);
        })

        .def_property_readonly("pcf", &TTimeSeries::pcf)
        .def_property_readonly("start_time", &TTimeSeries::start_time)
        .def_property_readonly("time_step", &TTimeSeries::time_step)
        .def_property_readonly("end_time", &TTimeSeries::end_time)
        .def_property_readonly("size", [](const TTimeSeries& self) { return self.size(); })

        .def("__eq__", &TTimeSeries::operator==)
        .def("__ne__", &TTimeSeries::operator!=)

        .def("__repr__", [](const TTimeSeries& self) {
          return self.to_string();
        })

        .def("__len__", [](const TTimeSeries& self) { return self.size(); })
        ;
    }
  };

}

void mpcf_py::register_timeseries(pybind11::module_& m)
{
  PyTimeSeriesBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
  PyTimeSeriesBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");
}
