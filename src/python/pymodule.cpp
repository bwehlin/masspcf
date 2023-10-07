#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <mpcf/pcf.h>
#include "pypcf_support.h"
#include <mpcf/algorithm.h>
#include <taskflow/taskflow.hpp>

#ifdef BUILD_WITH_CUDA
#include <mpcf/algorithms/cuda_matrix_integrate.h>
#endif

namespace py = pybind11;

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

#include <signal.h>



void catch_signals() {
  auto handler = [](int code) { throw std::runtime_error("SIGNAL " + std::to_string(code)); };
  signal(SIGINT, handler);
  signal(SIGTERM, handler);
  //signal(SIGKILL, handler);
}

template <typename F>
class Interruptable
{
public:
  Interruptable(F cmd)
    : m_cmd(cmd)
  { }

  auto run()
  {
    constexpr auto interruptCheckInterval = std::chrono::milliseconds(1000);

    py::gil_scoped_acquire acquire;

    tf::Executor exec(10);
    auto future = exec.async(m_cmd);
    std::future_status status(std::future_status::timeout);

    
    do
    {
      status = future.wait_for(interruptCheckInterval);
#if 0
      std::cout << "Check interrupt " << PyErr_CheckSignals() << std::endl;
      //Py_BEGIN_ALLOW_THREADS
      if (PyErr_CheckSignals() != 0)
      {
        std::cout << "Caugt signal" << std::endl;
        throw py::error_already_set();
      }
      //Py_END_ALLOW_THREADS
#endif
    } while (status != std::future_status::ready);

    return future.get();
  }

private:
  F m_cmd;
};

class Executor;

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
    m_status = std::move(rhs.m_status);
    m_future = std::move(rhs.m_future);
    return *this;
  }

  std::future_status wait_for(int timeoutMs)
  {
    std::cout << "Wait for " << timeoutMs << "ms" << std::endl;
    m_status = m_future.wait_for(std::chrono::milliseconds(timeoutMs));
    return m_status;
  }

  std::future_status get_last_status() const { return m_status; }

  auto get() { return m_future.get(); }

private:
  std::future<RetT> m_future;
  std::future_status m_status = std::future_status::deferred;
};

class Executor
{
public:

  tf::Executor& get()
  {
    return m_executor;
  }
  
  template <typename RetT>
  Future<RetT> async(std::function<RetT()> fn)
  {
    auto future = m_executor.async([=]() { return fn(); });
    return Future<RetT>(std::move(future));
    while (future.wait_for(std::chrono::milliseconds(1000)) != std::future_status::ready)
    {
      std::cout << "C++ wait on thread " << std::this_thread::get_id() << std::endl;
    }

    return Future<RetT>(std::move(future));
  }

private:
  tf::Executor m_executor;
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
    std::cout << "Called mem_avg on thread " << std::this_thread::get_id() << std::endl;
    return mpcf::mem_average(fs, chunksz);
#if 0
    return Interruptable([&]() { 
      return mpcf::mem_average(fs, chunksz); 
      }).run();
#endif
  }

  static std::future<mpcf::Pcf<Tt, Tv>> async_mem_average(const std::vector<mpcf::Pcf<Tt, Tv>>& fs, size_t chunksz)
  {
    static tf::Executor globalExec;
    //return std::async([&]() { return mpcf::mem_average(fs, chunksz); });
    std::future<mpcf::Pcf<Tt, Tv>> future;
    future = globalExec.async([&]() { return Interruptable([&]() {
      return mpcf::mem_average(fs, chunksz);
      }).run(); });
    return future;
  }

#if 0
  static BackgroundJob spawn_average(Executor& exec, const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
  {
    BackgroundJob job(exec, [&] { return mpcf::mem_average(fs, 2); });
    return job;
  }
#endif

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
    mpcf::matrix_l1_dist<Tt, Tv>(matrix.mutable_data(0), fs, mpcf::Executor::Cuda);
    return matrix;
#else
    mpcf::matrix_l1_dist<Tt, Tv>(matrix.mutable_data(0), fs, mpcf::Executor::Cpu);
    return matrix;
#endif
  }

  static Future<mpcf::Pcf<Tt, Tv>> spawn_pcf(Executor& e, std::function<mpcf::Pcf<Tt, Tv>()> fn)
  {
    return e.async<mpcf::Pcf<Tt, Tv>>(fn);
  }

  static Future<mpcf::Pcf<Tt, Tv>> spawn_avg(Executor& e, const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
  {
    return e.async<mpcf::Pcf<Tt, Tv>>([=]() { return mem_average(fs, 2); });
  }
};

template <typename Tt, typename Tv>
class PyBindings
{
public:
  void register_bindings(py::handle m, const std::string& suffix)
  {
    using TPcf = mpcf::Pcf<Tt, Tv>;

    py::class_<TPcf>(m, ("Pcf" + suffix).c_str())
      .def(py::init<>())
      .def(py::init<>([](py::array_t<Tt> arr){ return mpcf::detail::construct_pcf<Tt, Tv>(arr); }))
      .def("get_time_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tt); })
      .def("get_value_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tv); })
      .def("debug_print", &TPcf::debug_print) \
      .def("to_numpy", &mpcf::detail::to_numpy<TPcf>)
      .def("div_scalar", [](TPcf& self, Tv c){ return self /= c; })
      ;
    
    py::class_<Backend<Tt, Tv>> backend(m, ("Backend" + suffix).c_str());
    backend
      .def(py::init<>())
      .def_static("add", &Backend<Tt, Tv>::add)
      .def_static("combine", &Backend<Tt, Tv>::combine)
      .def_static("average", &Backend<Tt, Tv>::average)
      .def_static("mem_average", &Backend<Tt, Tv>::mem_average)
      .def_static("async_mem_average", &Backend<Tt, Tv>::async_mem_average)
      .def_static("spawn_pcf", &Backend<Tt, Tv>::spawn_pcf)
      .def_static("spawn_avg", &Backend<Tt, Tv>::spawn_avg)
      .def_static("st_average", &Backend<Tt, Tv>::st_average)
      .def_static("parallel_reduce", &Backend<Tt, Tv>::parallel_reduce)
      .def_static("l1_inner_prod", &Backend<Tt, Tv>::l1_inner_prod, py::return_value_policy::move)
      .def_static("matrix_l1_dist", &Backend<Tt, Tv>::matrix_l1_dist, py::return_value_policy::move)
      ;

    register_bindings_future<TPcf>(m, suffix, backend);
  }

private:
  template <typename RetT>
  void register_bindings_future(py::handle m, const std::string& suffix, py::class_<Backend<Tt, Tv>>& backend)
  {
    py::class_<Future<RetT>>(m, ("Future" + suffix).c_str())
      .def(py::init<>())
      .def("wait_for", &Future<RetT>::wait_for)
      .def("get", &Future<RetT>::get)
      .def("get_last_status", &Future<RetT>::get_last_status);

    //backend.def_static(("spawn_async" + suffix).c_str(), []())
  }
};

PYBIND11_MODULE(mpcf_cpp, m) {
  PyBindings<float, float>().register_bindings(m, "_f32_f32");
  PyBindings<double, double>().register_bindings(m, "_f64_f64");

  py::class_<Executor>(m, "Executor")
    .def(py::init<>());

  py::enum_<std::future_status>(m, "FutureStatus")
    .value("deferred", std::future_status::deferred)
    .value("ready", std::future_status::ready)
    .value("timeout", std::future_status::timeout)
    .export_values();
}
