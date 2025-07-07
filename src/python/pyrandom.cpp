/*
* Copyright 2024-2025 Bjorn Wehlin
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

#include <mpcf/random.h>
#include <mpcf/executor.h>

#include "pyarray.h"

#include <pybind11/pybind11.h>
#include <taskflow/algorithm/for_each.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <xtensor/xadapt.hpp>

namespace py = pybind11;

namespace
{
  template <typename Tt, typename Tv>
  class RandomBindings
  {
  public:
    static void noisy_sin(mpcf_py::NdArray<Tt, Tv>& out, size_t nPoints)
    {
      mpcf::noisy_function(out.data(), nPoints, [](Tv t) { return sin(2. * M_PI * t); });
    }

    static void noisy_cos(mpcf_py::NdArray<Tt, Tv>& out, size_t nPoints)
    {
      mpcf::noisy_function(out.data(), nPoints, [](Tv t) { return cos(2. * M_PI * t); });
    }
  };

  template <typename Tt, typename Tv>
  void register_types_random_bindings(py::handle m, const std::string& suffix)
  {
    using bindings = RandomBindings<Tt, Tv>;

    py::class_<bindings>(m, ("Random" + suffix).c_str())
      .def_static("noisy_sin", &bindings::noisy_sin)
      .def_static("noisy_cos", &bindings::noisy_cos);
  }

  using IntT = long long int;
  using FloatT = double;

  py::array_t<IntT> compute_random_weighted_samples(py::array_t<FloatT> probabilities, int nSamples, int sampleSize)
  {
    auto out = py::array_t<IntT>({static_cast<size_t>(probabilities.shape(0)), static_cast<size_t>(nSamples), static_cast<size_t>(sampleSize)});

    auto values = xt::arange(probabilities.shape(1));
    //auto weights = xt::zeros<FloatT>({probabilities.shape(1)});
    //auto v = xt::view(weights, xt::all(), xt::all());
    auto xProb = xt::adapt(probabilities.data(), xt::shape({probabilities.shape(0), probabilities.shape(1)}));

    tf::Taskflow flow;

    //for (auto i = 0; i < probabilities.shape(0); ++i)

    flow.for_each_index(size_t(0), size_t(probabilities.shape(0)), size_t(1),
        [&xProb, nSamples, sampleSize, &values, &out](size_t i) {
      auto curProbs = xt::strided_view(xProb, xt::xstrided_slice_vector({ i, xt::all() }));
      for (auto iSample = 0; iSample < nSamples; ++iSample)
      {
        auto chosen = xt::random::choice(values, sampleSize, curProbs, false, xt::random::get_default_random_engine());

        for (auto j = 0; j < sampleSize; ++j)
          out.template mutable_unchecked<3>()(i, iSample, j) = chosen(j);
      }
    });

    mpcf::default_executor().cpu()->run(std::move(flow));
    mpcf::default_executor().cpu()->wait_for_all();

    return out;
  }
}

void register_random_bindings(py::module_& m)
{
  register_types_random_bindings<float, float>(m, "_f32_f32");
  register_types_random_bindings<double, double>(m, "_f64_f64");

  m.def("compute_random_weighted_samples", &compute_random_weighted_samples);
}
