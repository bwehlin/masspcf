#include "py_subsample.hpp"

#include <sbear/sampling/subsample.hpp>

#include <pybind11/stl.h>  // std::variant caster

#include <string>
#include <utility>
#include <variant>

namespace py = pybind11;

namespace
{

  // The built-in functors accepted by the entry points below, bound as
  // descriptor classes. Adding a built-in = add the core functor, list it
  // here, and register its descriptor class in register_bindings.
  template <typename T>
  using FilterVariant = std::variant<sb::sampling::EuclideanDistance<T>>;
  template <typename T>
  using DistVariant = std::variant<sb::sampling::Gaussian<T>, sb::sampling::Uniform<T>>;

  template <typename T>
  class PySubsampleBindings
  {
  public:
    using TensorT = sb::Tensor<T>;
    using Gen = sb::DefaultRandomGenerator;

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PySubsampleBindings> cls(m, ("Subsample" + suffix).c_str());

      // Descriptor classes, nested under the precision namespace
      // (Subsample32.Gaussian etc.). The descriptors are the core functors
      // themselves; factory lambdas because the functors are aggregates.
      // Parameter validation lives in the Python spec classes.
      py::class_<sb::sampling::EuclideanDistance<T>>(cls, "Euclidean")
          .def(py::init<>());
      py::class_<sb::sampling::Gaussian<T>>(cls, "Gaussian")
          .def(py::init([](T mean, T sigma) { return sb::sampling::Gaussian<T>{mean, sigma}; }),
               py::arg("mean"), py::arg("sigma"));
      py::class_<sb::sampling::Uniform<T>>(cls, "Uniform")
          .def(py::init([](T low, T high) { return sb::sampling::Uniform<T>{low, high}; }),
               py::arg("low"), py::arg("high"));

      cls.def_static("sample_subsets",
          [](const TensorT& reference, const TensorT& query, const FilterVariant<T>& filter,
             const DistVariant<T>& distribution, size_t sampleSize, size_t nInstances,
             bool replace, const Gen* gen) {
            return std::visit([&](const auto& filterFunctor, const auto& distFunctor) {
              return to_tuple(sb::sampling::sample_subsets(
                  sb::PointCloud<T>(reference), sb::PointCloud<T>(query), filterFunctor,
                  distFunctor, sampleSize, nInstances, replace, resolve_generator(gen),
                  sb::default_executor()));
            }, filter, distribution);
          },
          py::arg("reference"), py::arg("query"), py::arg("filter"), py::arg("distribution"),
          py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
          py::arg("generator").none(true) = py::none());

      // Distance-matrix input: query is a tensor of reference row indices. The
      // "filter" is inherently the stored distance, so there is no filter
      // argument.
      cls.def_static("sample_subsets_distmat",
          [](const sb::DistanceMatrix<T>& source, const sb::Tensor<uint64_t>& query,
             const DistVariant<T>& distribution, size_t sampleSize, size_t nInstances,
             bool replace, const Gen* gen) {
            return std::visit([&](const auto& distFunctor) {
              return to_tuple(sb::sampling::sample_subsets_distmat(
                  source, query, distFunctor, sampleSize, nInstances, replace,
                  resolve_generator(gen), sb::default_executor()));
            }, distribution);
          },
          py::arg("source"), py::arg("query"), py::arg("distribution"),
          py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
          py::arg("generator").none(true) = py::none());
    }

  private:
    /// Resolve the nullable generator argument (Python None -> nullptr) to a
    /// concrete generator, falling back to the global default.
    static const Gen& resolve_generator(const Gen* gen)
    {
      return gen ? *gen : sb::default_generator();
    }

    template <typename ElemT>
    static py::tuple to_tuple(sb::sampling::SubsampleHandle<ElemT> handle)
    {
      return py::make_tuple(std::move(handle.task), std::move(handle.samples));
    }
  };

}

namespace sb_py
{
  void register_sampling_subsample(py::module_& m)
  {
    PySubsampleBindings<sb::float32_t>::register_bindings(m, "32");
    PySubsampleBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
