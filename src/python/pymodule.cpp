#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpcf/pcf.h>
#include "pypcf_support.h"
#include <mpcf/algorithm.h>

#ifdef BUILD_WITH_CUDA
#include <mpcf/algorithms/cuda_matrix_integrate.h>
#endif

namespace py = pybind11;

namespace
{
  struct Settings
  {
    bool forceCpu = false;
    
  } g_settings;
  
  template <typename Tt, typename Tv>
  class ReductionWrapper
  {
  public:
    using reduction_function = Tt(*)(Tt, Tt, Tv, Tv);
    ReductionWrapper(unsigned long long addr) : m_fn(reinterpret_cast<reduction_function>(addr)) { }
    Tt operator()(Tt left, Tt right, Tv top, Tv bottom) 
    {
      return m_fn(left, right, top, bottom);
    }
  private:
    reduction_function m_fn;
  };
  
  #define STRINGIFY(x) #x
  #define MACRO_STRINGIFY(x) STRINGIFY(x)
  
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
  
    static mpcf::Pcf<Tt, Tv> mem_average(const std::vector<mpcf::Pcf<Tt, Tv>>& fs, size_t chunksz)
    {
      return mpcf::mem_average(fs, chunksz);
    }
  
    static mpcf::Pcf<Tt, Tv> st_average(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      return mpcf::st_average(fs);
    }
  
    static mpcf::Pcf<Tt, Tv> parallel_reduce(const std::vector<mpcf::Pcf<Tt, Tv>>& fs, unsigned long long cb){ \
      ReductionWrapper<Tt, Tv> reduction(cb);
      return mpcf::parallel_reduce(fs, 
        [&reduction](const mpcf::Rectangle<Tt, Tv>& rect) -> Tt 
        { 
          return reduction(rect.left, rect.right, rect.top, rect.bottom); 
        });
    }
  
    static py::array_t<Tv> l1_inner_prod(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      py::array_t<Tv> matrix({fs.size(), fs.size()});
  #ifdef BUILD_WITH_CUDA
      mpcf::cuda_matrix_integrate<Tt, Tv>(matrix.mutable_data(0), fs, mpcf::device_ops::l1_inner_prod<Tt, Tv>());
  #else
      mpcf::matrix_integrate<Tt, Tv>(matrix.mutable_data(0), fs, 
        [](const typename mpcf::Pcf<Tt, Tv>::rectangle_type& rect){ 
          return (rect.right - rect.left) * rect.top * rect.bottom;
        });
  #endif
      return matrix;
    }
  
    static py::array_t<Tv> matrix_l1_dist(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      py::array_t<Tv> matrix({fs.size(), fs.size()});
  #ifdef BUILD_WITH_CUDA
      mpcf::Executor& exec = g_settings.forceCpu ? mpcf::default_cpu_executor() : mpcf::default_cuda_executor();
      mpcf::matrix_l1_dist<Tt, Tv>(matrix.mutable_data(0), fs, exec);
  #else
      mpcf::matrix_l1_dist<Tt, Tv>(matrix.mutable_data(0), fs, mpcf::default_cpu_executor());
  #endif
      return matrix;
    }
  };
  
  template <typename Tt, typename Tv>
  class PyBindings
  {
  public:
    void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<mpcf::Pcf<Tt, Tv>>(m, ("Pcf" + suffix).c_str())
        .def(py::init<>())
        .def(py::init<>([](py::array_t<Tt> arr){ return mpcf::detail::construct_pcf<Tt, Tv>(arr); }))
        .def("get_time_type", [](mpcf::Pcf<Tt, Tv>& /* self */) -> std::string { return STRINGIFY(Tt); })
        .def("get_value_type", [](mpcf::Pcf<Tt, Tv>& /* self */) -> std::string { return STRINGIFY(Tv); })
        .def("debug_print", &mpcf::Pcf<Tt, Tv>::debug_print) \
        .def("to_numpy", &mpcf::detail::to_numpy<mpcf::Pcf<Tt, Tv>>)
        .def("div_scalar", [](mpcf::Pcf<Tt, Tv>& self, Tv c){ return self /= c; })
        ;
      
      py::class_<Backend<Tt, Tv>>(m, ("Backend" + suffix).c_str())
        .def(py::init<>())
        .def_static("add", &Backend<Tt, Tv>::add)
        .def_static("combine", &Backend<Tt, Tv>::combine)
        .def_static("average", &Backend<Tt, Tv>::average)
        .def_static("mem_average", &Backend<Tt, Tv>::mem_average)
        .def_static("st_average", &Backend<Tt, Tv>::st_average)
        .def_static("parallel_reduce", &Backend<Tt, Tv>::parallel_reduce)
        .def_static("l1_inner_prod", &Backend<Tt, Tv>::l1_inner_prod, py::return_value_policy::move)
        .def_static("matrix_l1_dist", &Backend<Tt, Tv>::matrix_l1_dist, py::return_value_policy::move)
        ;
    }
  };

}

PYBIND11_MODULE(mpcf_cpp, m) {
  PyBindings<float, float>().register_bindings(m, "_f32_f32");
  PyBindings<double, double>().register_bindings(m, "_f64_f64");
  
  m.def("force_cpu", [](bool on){ g_settings.forceCpu = on; });
}
