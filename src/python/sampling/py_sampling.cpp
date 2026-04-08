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

#include "py_sampling.hpp"

#include <mpcf/sampling/distribution.hpp>
#include <mpcf/sampling/sampling_result.hpp>
#include <mpcf/sampling/sampler.hpp>

#include <pybind11/stl.h>

#include <limits>

namespace py = pybind11;

namespace
{

  template <typename T, typename SamplerT>
  void bind_sampler_sample(py::class_<SamplerT>& cls)
  {
    using PCloud = mpcf::PointCloud<T>;
    using Result = mpcf::sampling::SamplingResult<T>;
    using Gaussian = mpcf::sampling::Gaussian<T>;
    using Uniform = mpcf::sampling::Uniform<T>;
    using DefaultMixture = mpcf::sampling::DefaultMixture<T>;

    T inf = std::numeric_limits<T>::infinity();

    cls.def("sample", [](const SamplerT& self,
                          const PCloud& vantage,
                          size_t k,
                          py::object dist_obj,
                          bool replace,
                          const mpcf::DefaultRandomGenerator* gen,
                          T radius,
                          std::vector<T> stages,
                          size_t escalation_threshold,
                          size_t max_attempts) -> Result
        {
          auto do_sample = [&](const auto& d) -> Result {
            if (gen)
            {
              return self.sample(vantage, k, d, replace, *gen,
                                 radius, stages, escalation_threshold, max_attempts);
            }
            else
            {
              return self.sample(vantage, k, d, replace,
                                 mpcf::default_generator(),
                                 radius, stages, escalation_threshold, max_attempts);
            }
          };

          if (py::isinstance<Gaussian>(dist_obj))
            return do_sample(dist_obj.cast<const Gaussian&>());
          if (py::isinstance<Uniform>(dist_obj))
            return do_sample(dist_obj.cast<const Uniform&>());
          if (py::isinstance<DefaultMixture>(dist_obj))
            return do_sample(dist_obj.cast<const DefaultMixture&>());

          throw py::type_error("dist must be Gaussian, Uniform, or Mixture");
        },
        py::arg("vantage"), py::arg("k"), py::arg("dist"),
        py::arg("replace"),
        py::arg("generator").none(true) = py::none(),
        py::arg("radius") = inf,
        py::arg("stages") = std::vector<T>{T(0), T(0.1), T(0.5), T(1.0)},
        py::arg("escalation_threshold") = mpcf::sampling::default_escalation_threshold,
        py::arg("max_attempts") = mpcf::sampling::default_max_attempts);
  }

  template <typename T>
  class PySamplingBindings
  {
  public:
    using PCloud = mpcf::PointCloud<T>;
    using Collection = mpcf::sampling::IndexedPointCloudCollection<T>;
    using IPC = mpcf::sampling::IndexedPointCloud<T>;
    using TreeSamplerT = mpcf::sampling::TreeSampler<T>;
    using NaiveSamplerT = mpcf::sampling::NaiveSampler<T>;
    using Result = mpcf::sampling::SamplingResult<T>;
    using Diagnostics = mpcf::sampling::SamplingDiagnostics;

    using Gaussian = mpcf::sampling::Gaussian<T>;
    using Uniform = mpcf::sampling::Uniform<T>;
    using DefaultMixture = mpcf::sampling::DefaultMixture<T>;

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      // Distribution classes
      py::class_<Gaussian>(m, ("Gaussian" + suffix).c_str())
          .def(py::init<T, T>(), py::arg("mean"), py::arg("sigma"))
          .def("__call__", &Gaussian::operator(), py::arg("d"))
          .def("max", &Gaussian::max)
          .def("max_in_range", &Gaussian::max_in_range, py::arg("d_min"), py::arg("d_max"))
          .def_readwrite("mean", &Gaussian::mean)
          .def_readwrite("sigma", &Gaussian::sigma);

      py::class_<Uniform>(m, ("Uniform" + suffix).c_str())
          .def(py::init<T, T>(), py::arg("lo"), py::arg("hi"))
          .def("__call__", &Uniform::operator(), py::arg("d"))
          .def("max", &Uniform::max)
          .def("max_in_range", &Uniform::max_in_range, py::arg("d_min"), py::arg("d_max"))
          .def_readwrite("lo", &Uniform::lo)
          .def_readwrite("hi", &Uniform::hi);

      using Component = typename DefaultMixture::component_type;

      py::class_<DefaultMixture>(m, ("Mixture" + suffix).c_str())
          .def(py::init([](py::list py_components, std::vector<T> weights) {
            std::vector<Component> components;
            components.reserve(py::len(py_components));
            for (const auto& obj : py_components)
            {
              if (py::isinstance<Gaussian>(obj))
                components.push_back(obj.cast<Gaussian>());
              else if (py::isinstance<Uniform>(obj))
                components.push_back(obj.cast<Uniform>());
              else
                throw py::type_error("Mixture components must be Gaussian or Uniform");
            }
            return DefaultMixture{std::move(components), std::move(weights)};
          }), py::arg("components"), py::arg("weights"))
          .def("__call__", &DefaultMixture::operator(), py::arg("x"))
          .def("max", &DefaultMixture::max)
          .def("max_in_range", &DefaultMixture::max_in_range, py::arg("x_min"), py::arg("x_max"))
          .def_readwrite("weights", &DefaultMixture::weights);

      // IndexedPointCloud
      py::class_<IPC>(m, ("IndexedPointCloud" + suffix).c_str())
          .def("n_points", &IPC::n_points)
          .def("dim", &IPC::dim)
          .def("materialize", &IPC::materialize)
          .def_property_readonly("indices", &IPC::indices);

      // IndexedPointCloudCollection
      py::class_<Collection>(m, ("IndexedPointCloudCollection" + suffix).c_str())
          .def("__len__", &Collection::n_vantage)
          .def("__getitem__", &Collection::operator[])
          .def("n_vantage", &Collection::n_vantage)
          .def("k", &Collection::k)
          .def("dim", &Collection::dim)
          .def_property_readonly("source", &Collection::source)
          .def_property_readonly("indices", &Collection::indices);

      // SamplingResult
      py::class_<Result>(m, ("SamplingResult" + suffix).c_str())
          .def_readonly("collection", &Result::collection)
          .def_readonly("diagnostics", &Result::diagnostics);

      // TreeSampler
      auto tree_cls = py::class_<TreeSamplerT>(m, ("TreeSampler" + suffix).c_str())
          .def(py::init<PCloud>(), py::arg("source"))
          .def_property_readonly("dim", &TreeSamplerT::dim)
          .def_property_readonly("n_points", &TreeSamplerT::n_points);
      bind_sampler_sample<T>(tree_cls);

      // NaiveSampler
      auto naive_cls = py::class_<NaiveSamplerT>(m, ("NaiveSampler" + suffix).c_str())
          .def(py::init<PCloud>(), py::arg("source"))
          .def_property_readonly("dim", &NaiveSamplerT::dim)
          .def_property_readonly("n_points", &NaiveSamplerT::n_points);
      bind_sampler_sample<T>(naive_cls);

    }
  };

}

void mpcf_py::register_sampling(py::module_& m)
{
  using Diagnostics = mpcf::sampling::SamplingDiagnostics;

  py::class_<Diagnostics>(m, "SamplingDiagnostics")
      .def_readonly("acceptance_rate", &Diagnostics::acceptance_rate)
      .def_readonly("total_attempts", &Diagnostics::total_attempts)
      .def_readonly("biased", &Diagnostics::biased)
      .def("all_exact", &Diagnostics::all_exact)
      .def("biased_vantage_count", &Diagnostics::biased_vantage_count);

  PySamplingBindings<mpcf::float32_t>::register_bindings(m, "32");
  PySamplingBindings<mpcf::float64_t>::register_bindings(m, "64");
}
