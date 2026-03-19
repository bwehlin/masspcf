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

#include "py_pcf.h"
#include "../py_future.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpcf/functional/pcf.h>
#include "../pypcf_support.h"
#include <mpcf/algorithm.h>
#include <mpcf/executor.h>
#include <mpcf/task.h>

#include "../py_np_support.h"
#include "../py_settings.h"

#include <iostream>

namespace py = pybind11;

namespace
{

#define STRINGIFY(x) #x

  template <typename Tt, typename Tv>
  class ReductionWrapper
  {
  public:
    using reduction_function = Tt(*)(Tt, Tt, Tv, Tv);
    explicit ReductionWrapper(unsigned long long addr) : m_fn(reinterpret_cast<reduction_function>(addr)) { }
    Tt operator()(Tt left, Tt right, Tv top, Tv bottom)
    {
      return m_fn(left, right, top, bottom);
    }
  private:
    reduction_function m_fn;
  };

  template <typename Tt, typename Tv>
  class Backend
  {
  public:
    static mpcf::Pcf<Tt, Tv> add(const mpcf::Pcf<Tt, Tv>& f, const mpcf::Pcf<Tt, Tv>& g)
    {
      return f + g;
    }

    static mpcf::Pcf<Tt, Tv> combine(const mpcf::Pcf<Tt, Tv>& f, const mpcf::Pcf<Tt, Tv>& g, unsigned long long cb)
    {
      ReductionWrapper<Tt, Tv> reduction(cb);
      return mpcf::combine(f, g,
        [&reduction](const mpcf::Rectangle<Tt, Tv>& rect) -> Tt {
          return reduction(rect.left, rect.right, rect.top, rect.bottom);
        });
    }

    static mpcf::Pcf<Tt, Tv> average(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      return mpcf::average(fs);
    }

    static mpcf::Pcf<Tt, Tv> parallel_reduce(const std::vector<mpcf::Pcf<Tt, Tv>>& fs, unsigned long long cb){ \
      ReductionWrapper<Tt, Tv> reduction(cb);
      return mpcf::parallel_reduce(fs.begin(), fs.end(),
        [&reduction](const mpcf::Rectangle<Tt, Tv>& rect) -> Tt
        {
          return reduction(rect.left, rect.right, rect.top, rect.bottom);
        });
    }

    static Tv single_l1_norm(const mpcf::Pcf<Tt, Tv>& f)
    {
      return mpcf::l1_norm(f);
    }

    static Tv single_l2_norm(const mpcf::Pcf<Tt, Tv>& f)
    {
      return mpcf::l2_norm(f);
    }

    static Tv single_lp_norm(const mpcf::Pcf<Tt, Tv>& f, /* let's stick with float64_t here to make life a bit easier */ mpcf::float64_t p)
    {
      return mpcf::lp_norm(f, Tv(p));
    }

    static Tv single_linfinity_norm(const mpcf::Pcf<Tt, Tv>& f)
    {
      return mpcf::linfinity_norm(f);
    }

    template <typename TOperation, typename PcfFwdIt>
    static std::unique_ptr<mpcf::StoppableTask<void>> matrix_integrate(py::array_t<Tv>& matrix, PcfFwdIt beginPcfs, PcfFwdIt endPcfs, TOperation op)
    {
      auto* out = matrix.mutable_data(0);

      if (mpcf_py::g_settings.deviceVerbose)
      {
        std::cout << "Integral computation on CPU(s)" << std::endl;
      }

      auto task = std::make_unique<mpcf::MatrixIntegrateCpuTask<TOperation, PcfFwdIt>>(out, beginPcfs, endPcfs, op);
      task->start_async(mpcf::default_executor());
      return task;
    }
  };

  template <typename Tt, typename Tv>
  class PyBindings
  {
  public:
    static void register_bindings(py::handle m, const std::string& suffix)
    {
      using TPcf = mpcf::Pcf<Tt, Tv>;
      using point_type = typename TPcf::point_type;

      py::class_<mpcf::Pcf<Tt, Tv>>(m, ("Pcf" + suffix).c_str(), py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<>([](py::array_t<Tt> arr){ return mpcf::detail::construct_pcf<Tt, Tv>(arr); }))
        .def("get_time_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tt); })
        .def("get_value_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tv); })
        .def("debug_print", &TPcf::debug_print) \
        .def_buffer([](TPcf& self) { return mpcf::detail::to_numpy<mpcf::Pcf<Tt, Tv>>(self); })
        .def("size", [](const TPcf& self){ return self.points().size(); })
        .def("copy", [](const TPcf& self){ return TPcf(self); })
        .def("__add__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self + rhs; })
        .def("__add__", [](const TPcf& self, Tv c) -> TPcf { return self + c; })
        .def("__radd__", [](const TPcf& self, Tv c) -> TPcf { return c + self; })
        .def("__sub__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self - rhs; })
        .def("__sub__", [](const TPcf& self, Tv c) -> TPcf { return self - c; })
        .def("__rsub__", [](const TPcf& self, Tv c) -> TPcf { return c - self; })
        .def("__mul__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self * rhs; })
        .def("__mul__", [](const TPcf& self, Tv c) -> TPcf { return self * c; })
        .def("__rmul__", [](const TPcf& self, Tv c) -> TPcf { return c * self; })
        .def("__truediv__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self / rhs; })
        .def("__truediv__", [](const TPcf& self, Tv c) -> TPcf { return self / c; })
        .def("__rtruediv__", [](const TPcf& self, Tv c) -> TPcf { return c / self; })
        .def("__neg__", [](const TPcf& self) -> TPcf { return -self; })
        .def("__pow__", [](const TPcf& self, Tv c) -> TPcf {
          auto result = mpcf::pow(self, c);
          for (const auto& pt : result.points())
          {
            if (std::isnan(pt.v) || std::isinf(pt.v))
            {
              PyErr_WarnEx(PyExc_RuntimeWarning,
                "invalid or infinite value encountered in pcf pow", 1);
              break;
            }
          }
          return result;
        })
        .def("__call__", [](const TPcf& self, Tt t) -> Tv { return self.evaluate(t); })
        .def("__call__", [](const TPcf& self, py::array_t<Tt> times) -> py::array_t<Tv> {
          // Flatten to 1D for processing, remember original shape
          auto original_shape = std::vector<py::ssize_t>(times.shape(), times.shape() + times.ndim());
          auto flat_times = times.reshape({times.size()});
          NumpyTensor<Tt> in(flat_times);
          auto n = static_cast<size_t>(flat_times.size());

          // Argsort
          std::vector<size_t> order(n);
          std::iota(order.begin(), order.end(), 0);
          std::sort(order.begin(), order.end(), [&in](size_t a, size_t b) {
            return in(a) < in(b);
          });

          // Build sorted times and evaluate
          py::array_t<Tt> sorted_times({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tt> sorted_in(sorted_times);
          for (size_t i = 0; i < n; ++i)
          {
            sorted_in(i) = in(order[i]);
          }

          py::array_t<Tv> sorted_result({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tv> sorted_out(sorted_result);
          self.evaluate(sorted_in, sorted_out, n);

          // Unsort results back to original order
          py::array_t<Tv> flat_result({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tv> out(flat_result);
          for (size_t i = 0; i < n; ++i)
          {
            out(order[i]) = sorted_out(i);
          }

          return flat_result.reshape(original_shape);
        })

        // We (un-)pickle the raw bytes stored in points()
        .def(py::pickle([](TPcf& self) {
            const unsigned char* data = reinterpret_cast<const unsigned char*>(self.points().data());
            std::vector<unsigned char> buf;
            buf.resize(sizeof(point_type) * self.points().size());
            memcpy(buf.data(), data, buf.size());
            return buf;
          }, [](const std::vector<unsigned char>& t){
            std::vector<point_type> pts;
            pts.resize(t.size() / sizeof(point_type));
            memcpy(pts.data(), t.data(), t.size());
            TPcf f(std::move(pts));
            return f;
          }))
        ;

      py::class_<Backend<Tt, Tv>> backend(m, ("Backend" + suffix).c_str());
      backend
        .def(py::init<>())
        .def_static("add", &Backend<Tt, Tv>::add)
        .def_static("combine", &Backend<Tt, Tv>::combine)
        .def_static("average", &Backend<Tt, Tv>::average)
        .def_static("parallel_reduce", &Backend<Tt, Tv>::parallel_reduce)

        .def_static("single_l1_norm", &Backend<Tt, Tv>::single_l1_norm)
        .def_static("single_l2_norm", &Backend<Tt, Tv>::single_l2_norm)
        .def_static("single_lp_norm", &Backend<Tt, Tv>::single_lp_norm)
        .def_static("single_linfinity_norm", &Backend<Tt, Tv>::single_linfinity_norm)

      /*
        .def_static("list_l1_norm", &Backend<Tt, Tv>::list_l1_norm)
        .def_static("list_l2_norm", &Backend<Tt, Tv>::list_l2_norm)
        //.def_static("list_lp_norm", &Backend<Tt, Tv>::list_lp_norm)
        .def_static("list_linfinity_norm", &Backend<Tt, Tv>::list_linfinity_norm)

        .def_static("calc_pdist_1", &Backend<Tt, Tv>::pdist_1)
        .def_static("calc_pdist_p", &Backend<Tt, Tv>::pdist_p)
        .def_static("calc_l2_kernel", &Backend<Tt, Tv>::l2_kernel)
        */
        ;

      mpcf_py::register_bindings_future<TPcf>(m, suffix);
    }
  };

}

void mpcf_py::register_pcf(pybind11::module_& m)
{
  PyBindings<mpcf::float32_t, mpcf::float32_t>::register_bindings(m, "_f32_f32");
  PyBindings<mpcf::float64_t, mpcf::float64_t>::register_bindings(m, "_f64_f64");
  PyBindings<mpcf::int32_t, mpcf::int32_t>::register_bindings(m, "_i32_i32");
  PyBindings<mpcf::int64_t, mpcf::int64_t>::register_bindings(m, "_i64_i64");
}
