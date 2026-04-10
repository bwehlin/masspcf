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

#include <sstream>

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

        .def(py::init([](py::array_t<Tt> values, Tt start_time, Tt time_step) {
          py::buffer_info buf = values.request();
          if (buf.ndim == 1)
          {
            // 1-D array of values: breakpoints at 0, 1, 2, ...
            auto data = values.template unchecked<1>();
            std::vector<mpcf::TimePoint<Tt, Tv>> points;
            points.reserve(buf.shape[0]);
            for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
            {
              points.emplace_back(static_cast<Tt>(i), static_cast<Tv>(data(i)));
            }
            return TTimeSeries(TPcf(std::move(points)), start_time, time_step);
          }
          else if (buf.ndim == 2 && buf.shape[1] == 2)
          {
            // (n, 2) array of (time, value) pairs: epoch = first time
            auto data = values.template unchecked<2>();
            Tt first_time = data(0, 0);
            std::vector<mpcf::TimePoint<Tt, Tv>> points;
            points.reserve(buf.shape[0]);
            for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
            {
              Tt pcf_t = (data(i, 0) - first_time) / time_step;
              points.emplace_back(pcf_t, static_cast<Tv>(data(i, 1)));
            }
            return TTimeSeries(TPcf(std::move(points)), first_time, time_step);
          }
          else
          {
            throw py::value_error(
              "Expected a 1-D array of values or a (n, 2) array of (time, value) pairs.");
          }
        }), py::arg("values"), py::arg("start_time") = Tt(0), py::arg("time_step") = Tt(1))

        .def("__call__", [](const TTimeSeries& self, Tt t) -> Tv {
          return self.evaluate(t);
        })

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
