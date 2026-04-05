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

#ifndef MPCF_SAMPLING_DISTRIBUTION_H
#define MPCF_SAMPLING_DISTRIBUTION_H

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numeric>
#include <variant>
#include <vector>

namespace mpcf::sampling
{

  /// A weight function that maps a non-negative scalar to a non-negative weight.
  /// Must support evaluation, a global maximum, and a tight maximum over an interval.
  template <typename D, typename T>
  concept WeightFunction = requires(const D& d, T val)
  {
    { d(val) } -> std::convertible_to<T>;
    { d.max() } -> std::convertible_to<T>;
    { d.max_in_range(val, val) } -> std::convertible_to<T>;
  };

  template <typename T>
  struct Gaussian
  {
    T mean;
    T sigma;

    T operator()(T d) const
    {
      T z = (d - mean) / sigma;
      return std::exp(T(-0.5) * z * z);
    }

    T max() const
    {
      return T(1);
    }

    T max_in_range(T d_min, T d_max) const
    {
      // The peak is at d = mean. If mean is in [d_min, d_max], max is 1.
      if (d_min <= mean && mean <= d_max)
        return T(1);

      // Otherwise, the closer endpoint gives the larger value.
      return std::max((*this)(d_min), (*this)(d_max));
    }
  };

  template <typename T>
  struct Uniform
  {
    T lo;
    T hi;

    T operator()(T d) const
    {
      return (d >= lo && d <= hi) ? T(1) : T(0);
    }

    T max() const
    {
      return T(1);
    }

    T max_in_range(T d_min, T d_max) const
    {
      // If the intervals overlap, the max is 1; otherwise 0.
      if (d_max < lo || d_min > hi)
        return T(0);
      return T(1);
    }
  };

  template <typename T, typename... Components>
  struct Mixture
  {
    using component_type = std::variant<Components...>;

    std::vector<component_type> components;
    std::vector<T> weights;

    T operator()(T x) const
    {
      T result = T(0);
      for (size_t i = 0; i < components.size(); ++i)
      {
        result += weights[i] * std::visit([x](const auto& c) { return c(x); }, components[i]);
      }
      return result;
    }

    T max() const
    {
      T result = T(0);
      for (size_t i = 0; i < components.size(); ++i)
      {
        result += weights[i] * std::visit([](const auto& c) { return c.max(); }, components[i]);
      }
      return result;
    }

    T max_in_range(T x_min, T x_max) const
    {
      T result = T(0);
      for (size_t i = 0; i < components.size(); ++i)
      {
        result += weights[i] * std::visit([x_min, x_max](const auto& c) {
          return c.max_in_range(x_min, x_max);
        }, components[i]);
      }
      return result;
    }
  };

  /// Default mixture type supporting all built-in weight functions.
  template <typename T>
  using DefaultMixture = Mixture<T, Gaussian<T>, Uniform<T>>;

}

#endif
