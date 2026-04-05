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
#include <mpcf/sampling/indexed_point_cloud.hpp>
#include <mpcf/sampling/sampler.hpp>

#include <pybind11/stl.h>

#include <limits>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PySamplingBindings
  {
  public:
    using PCloud = mpcf::PointCloud<T>;
    using Collection = mpcf::sampling::IndexedPointCloudCollection<T>;
    using IPC = mpcf::sampling::IndexedPointCloud<T>;
    using SamplerT = mpcf::sampling::DistanceWeightedSampler<T>;

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

      T inf = std::numeric_limits<T>::infinity();

      // DistanceWeightedSampler
      py::class_<SamplerT>(m, ("DistanceWeightedSampler" + suffix).c_str())
          .def(py::init<PCloud>(), py::arg("source"))
          .def("sample", [](const SamplerT& self,
                            const PCloud& vantage,
                            size_t k,
                            py::object dist_obj,
                            bool replace,
                            const mpcf::DefaultRandomGenerator* gen,
                            T radius,
                            T min_correction) -> Collection
          {
            auto do_sample = [&](const auto& d) -> Collection {
              if (gen)
              {
                return self.sample(vantage, k, d, replace, *gen,
                                   mpcf::default_executor(), radius, min_correction);
              }
              else
              {
                return self.sample(vantage, k, d, replace,
                                   mpcf::default_generator(), mpcf::default_executor(),
                                   radius, min_correction);
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
          py::arg("min_correction") = T(0))
          .def_property_readonly("dim", &SamplerT::dim)
          .def_property_readonly("n_points", &SamplerT::n_points);
    }
  };

}

void mpcf_py::register_sampling(py::module_& m)
{
  PySamplingBindings<mpcf::float32_t>::register_bindings(m, "32");
  PySamplingBindings<mpcf::float64_t>::register_bindings(m, "64");
}
