#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpcf/pcf.h>
#include "pypcf_support.h"
#include <mpcf/algorithms/cuda_matrix_integrate.h>

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

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DECLARE_PCF(Tt, Tv, name) \
  py::class_<mpcf::Pcf<Tt, Tv>>(m, STRINGIFY(name)) \
    .def(py::init<>()) \
    .def(py::init<>([](py::array_t<mpcf::Pcf<Tt, Tv>::time_type> arr){ return mpcf::detail::construct_pcf<Tt, Tv>(arr); })) \
    .def("get_time_type", [](mpcf::Pcf<Tt, Tv>& /* self */) -> std::string { return STRINGIFY(Tt); }) \
    .def("get_value_type", [](mpcf::Pcf<Tt, Tv>& /* self */) -> std::string { return STRINGIFY(Tv); }) \
    .def("debugPrint", &mpcf::Pcf<Tt, Tv>::debugPrint) \
    .def("to_numpy", &mpcf::detail::to_numpy<mpcf::Pcf<Tt, Tv>>) \
    .def("div_scalar", [](mpcf::Pcf<Tt, Tv>& self, Tv c){ return self /= c; }) \
    ; \
  m.def(STRINGIFY(name##_add), [](const mpcf::Pcf<Tt, Tv>& f, const mpcf::Pcf<Tt, Tv>& g){ return f + g; }); \
  m.def(STRINGIFY(name##_combine), \
  [](const mpcf::Pcf<Tt, Tv>& f, const mpcf::Pcf<Tt, Tv>& g, unsigned long long cb){ \
    ReductionWrapper<Tt, Tv> reduction(cb); \
    return mpcf::combine(f, g, [&reduction](const mpcf::Rectangle<Tt, Tv>& rect) -> Tt { return reduction(rect.left, rect.right, rect.top, rect.bottom); }); \
  }); \
  m.def(STRINGIFY(name##_average), [](const std::vector<mpcf::Pcf<Tt, Tv>>& fs){ return mpcf::average(fs); }); \
  m.def(STRINGIFY(name##_parallel_reduce), \
  [](const std::vector<mpcf::Pcf<Tt, Tv>>& fs, unsigned long long cb){ \
    ReductionWrapper<Tt, Tv> reduction(cb); \
    return mpcf::parallel_reduce(fs, [&reduction](const mpcf::Rectangle<Tt, Tv>& rect) -> Tt { return reduction(rect.left, rect.right, rect.top, rect.bottom); }); \
  }); \
  m.def(STRINGIFY(name##_l1_inner_prod), [](const std::vector<mpcf::Pcf<Tt, Tv>>& fs) -> py::array_t<Tv> { \
    py::array_t<Tv> matrix({fs.size(), fs.size()}); \
    mpcf::matrix_integrate<Tt, Tv>(matrix.mutable_data(0), fs, mpcf::device_ops::l1_inner_prod<Tt, Tv>()); \
    return matrix; \
  }, py::return_value_policy::move);

#define DECLARE_RECTANGLE(Tt, Tv, name) \
  py::class_<Rectangle<Tt, Tv>>(m, STRINGIFY(name)) \
    .def(py::init<>()) \
    .def(py::init<Tt, Tt, Tv, Tv>()) \
    .def_property("left", [](Rectangle<Tt, Tv>& self){ return self.left; }, [](Rectangle<Tt, Tv>& self, Tv val){ self.left = val; } ) \
    .def_property("right", [](Rectangle<Tt, Tv>& self){ return self.right; }, [](Rectangle<Tt, Tv>& self, Tv val){ self.right = val; } ) \
    .def_property("top", [](Rectangle<Tt, Tv>& self){ return self.top; }, [](Rectangle<Tt, Tv>& self, Tv val){ self.top = val; } ) \
    .def_property("bottom", [](Rectangle<Tt, Tv>& self){ return self.bottom; }, [](Rectangle<Tt, Tv>& self, Tv val){ self.bottom = val; } ) \
    ;

PYBIND11_MODULE(mpcf_py, m) {
  DECLARE_PCF(float, float, Pcf_f32_f32)
//  DECLARE_RECTANGLE(float, float, Rectangle_f32_f32)

  DECLARE_PCF(double, double, Pcf_f64_f64)
//  DECLARE_RECTANGLE(double, double, Rectangle_f64_f64)
}
