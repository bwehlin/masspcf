#include <future>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpcf/pcf.h>
#include "pypcf_support.h"
#include <mpcf/algorithm.h>
#include <mpcf/executor.h>
#include <mpcf/task.h>
//#include <mpcf/array.h>

#ifdef BUILD_WITH_CUDA
#include <mpcf/cuda/cuda_matrix_integrate.cuh>
#endif

#include <iostream>

namespace py = pybind11;

namespace
{
  struct Settings
  {
    bool forceCpu = false;

#ifdef BUILD_WITH_CUDA
    dim3 blockDim = dim3(1, 32, 1);
#endif
    
  } g_settings;

  template <typename RetT>
  class Future
  {
  public:
    Future() = default;
    Future(std::future<RetT>&& future)
      : m_future(std::move(future))
    {
  
    }
  
    Future(const Future&) = delete;
    Future(Future&& other) noexcept
      : m_future(std::move(other.m_future))
    { }
  
    Future& operator=(const Future&) = delete;
    Future& operator=(Future&& rhs) noexcept
    {
      m_future = std::move(rhs.m_future);
      return *this;
    }
  
    std::future_status wait_for(int timeoutMs)
    {
      return m_future.wait_for(std::chrono::milliseconds(timeoutMs));
    }
  
    auto get()
    {
      if constexpr (!std::is_same_v<RetT, void>)
      {
        return m_future.get();
      }
      else
      {
        m_future.get();
      }
    }


  
  private:
    std::future<RetT> m_future;
  };
  
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
  
    static mpcf::Pcf<Tt, Tv> parallel_reduce(const std::vector<mpcf::Pcf<Tt, Tv>>& fs, unsigned long long cb){ \
      ReductionWrapper<Tt, Tv> reduction(cb);
      return mpcf::parallel_reduce(fs, 
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

    static Tv single_lp_norm(const mpcf::Pcf<Tt, Tv>& f, /* let's stick with double here to make life a bit easier */ double p)
    {
      return mpcf::lp_norm(f, Tv(p));
    }

    static Tv single_linfinity_norm(const mpcf::Pcf<Tt, Tv>& f)
    {
      return mpcf::linfinity_norm(f);
    }

    static void list_l1_norm(py::array_t<Tv>& out, std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      auto* outdata = out.mutable_data(0);
      mpcf::apply_functional(fs.begin(), fs.end(), outdata, mpcf::l1_norm<mpcf::Pcf<Tt, Tv>>);
    }
    
    static void list_l2_norm(py::array_t<Tv>& out, std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      auto* outdata = out.mutable_data(0);
      mpcf::apply_functional(fs.begin(), fs.end(), outdata, mpcf::l2_norm<mpcf::Pcf<Tt, Tv>>);
    }

    static void list_linfinity_norm(py::array_t<Tv>& out, std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      auto* outdata = out.mutable_data(0);
      mpcf::apply_functional(fs.begin(), fs.end(), outdata, mpcf::linfinity_norm<mpcf::Pcf<Tt, Tv>>);
    }

    static std::unique_ptr<mpcf::StoppableTask<void>> matrix_l1_dist(py::array_t<Tv>& matrix, std::vector<mpcf::Pcf<Tt, Tv>>& fs)
    {
      auto* out = matrix.mutable_data(0);

#ifdef BUILD_WITH_CUDA
      if (!g_settings.forceCpu)
      {
        auto task = mpcf::create_matrix_l1_distance_cuda_task<Tt, Tv>(out, std::move(fs), 0., std::numeric_limits<Tv>::max());
        task->set_block_dim(g_settings.blockDim);
        task->start_async(mpcf::default_executor());
        return task;
      }
#endif
      auto task = std::make_unique<mpcf::MatrixL1DistCpuTask<Tt, Tv>>(out, std::move(fs));
      task->start_async(mpcf::default_executor());
      return task;
    }
  };
  
  template <typename RetT>
  static void register_bindings_future(py::handle m, const std::string& suffix)
  {
    py::class_<Future<RetT>>(m, ("Future" + suffix).c_str())
      .def(py::init<>())
      .def("wait_for", &Future<RetT>::wait_for);
  }
  
  template <typename RetT>
  static void register_bindings_stoppable_task(py::handle m, const std::string& suffix)
  {
    py::class_<mpcf::StoppableTask<RetT>> cls(m, ("StoppableTask" + suffix).c_str());
    
    cls
        .def("request_stop", &mpcf::StoppableTask<RetT>::request_stop)
        .def("wait_for", [](mpcf::StoppableTask<RetT>& self, int ms) { return self.wait_for(std::chrono::milliseconds(ms)); })
        .def("work_total", &mpcf::StoppableTask<RetT>::work_total)
        .def("work_completed", &mpcf::StoppableTask<RetT>::work_completed)
        .def("work_step", &mpcf::StoppableTask<RetT>::work_step)
        .def("work_step_desc", &mpcf::StoppableTask<RetT>::work_step_desc)
        .def("work_step_unit", &mpcf::StoppableTask<RetT>::work_step_unit)
    ;
  }
  
  template <typename Tt, typename Tv>
  class PyBindings
  {
  public:
    static void register_bindings(py::handle m, const std::string& suffix)
    {
      using TPcf = mpcf::Pcf<Tt, Tv>;
      using point_type = typename TPcf::point_type;
      
      py::class_<mpcf::Pcf<Tt, Tv>>(m, ("Pcf" + suffix).c_str())
        .def(py::init<>())
        .def(py::init<>([](py::array_t<Tt> arr){ return mpcf::detail::construct_pcf<Tt, Tv>(arr); }))
        .def("get_time_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tt); })
        .def("get_value_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tv); })
        .def("debug_print", &TPcf::debug_print) \
        .def("to_numpy", &mpcf::detail::to_numpy<mpcf::Pcf<Tt, Tv>>)
        .def("div_scalar", [](TPcf& self, Tv c){ return self /= c; })
        .def("size", [](const TPcf& self){ return self.points().size(); })
        .def("copy", [](const TPcf& self){ return TPcf(self); })
          
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

        .def_static("matrix_l1_dist", &Backend<Tt, Tv>::matrix_l1_dist, py::return_value_policy::move)

        .def_static("single_l1_norm", &Backend<Tt, Tv>::single_l1_norm)
        .def_static("single_l2_norm", &Backend<Tt, Tv>::single_l2_norm)
        .def_static("single_lp_norm", &Backend<Tt, Tv>::single_lp_norm)
        .def_static("single_linfinity_norm", &Backend<Tt, Tv>::single_linfinity_norm)

        .def_static("list_l1_norm", &Backend<Tt, Tv>::list_l1_norm)
        .def_static("list_l2_norm", &Backend<Tt, Tv>::list_l2_norm)
        //.def_static("list_lp_norm", &Backend<Tt, Tv>::list_lp_norm)
        .def_static("list_linfinity_norm", &Backend<Tt, Tv>::list_linfinity_norm)
        ;
      
      register_bindings_future<TPcf>(m, suffix);
    }
  };
  
  

}

PYBIND11_MODULE(mpcf_cpp, m) {
  PyBindings<float, float>::register_bindings(m, "_f32_f32");
  PyBindings<double, double>::register_bindings(m, "_f64_f64");
  
  register_bindings_stoppable_task<void>(m, "_void");
  
  py::enum_<std::future_status>(m, "FutureStatus")
    .value("deferred", std::future_status::deferred)
    .value("ready", std::future_status::ready)
    .value("timeout", std::future_status::timeout)
    .export_values();

  py::class_<Future<void>>(m, "Future_void")
    .def(py::init<>())
    .def("wait_for", &Future<void>::wait_for);
  
  m.def("force_cpu", [](bool on){ g_settings.forceCpu = on; });
#ifdef BUILD_WITH_CUDA
  m.def("set_block_dim", [](unsigned int x, unsigned int y) { g_settings.blockDim = dim3(x, y, 1); });
  m.def("limit_gpus", [](size_t n){ mpcf::default_executor().limit_cuda_workers(n); });
#endif
  
  m.def("limit_cpus", [](size_t n){ mpcf::default_executor().limit_cpu_workers(n); });
  
}
