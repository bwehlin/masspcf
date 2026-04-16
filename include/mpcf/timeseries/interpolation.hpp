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

#ifndef MPCF_TIMESERIES_INTERPOLATION_H
#define MPCF_TIMESERIES_INTERPOLATION_H

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

namespace mpcf
{
  /// Built-in interpolation modes; used as the public setter API and as the
  /// on-disk serialization tag.
  enum class InterpolationMode : uint8_t { Nearest = 0, Linear = 1 };

  /// Default value used when a query falls outside the time series domain.
  /// Arithmetic types get quiet NaN; object types get a default-constructed
  /// instance. Specialize for types with a more meaningful sentinel.
  template <typename Tv, typename Enable = void>
  struct OutOfDomainValue
  {
    static Tv get() { return Tv{}; }
  };

  template <typename Tv>
  struct OutOfDomainValue<Tv, std::enable_if_t<std::is_arithmetic_v<Tv>>>
  {
    static Tv get() { return std::numeric_limits<Tv>::quiet_NaN(); }
  };

  /// Concept: `Tv` supports the arithmetic needed for linear blending with
  /// `Tt` weights — `(Tt * Tv) -> Tv` and `(Tv + Tv) -> Tv`.
  template <typename Tt, typename Tv>
  concept LinearlyBlendable = requires(Tt a, Tv v)
  {
    { a * v } -> std::convertible_to<Tv>;
    { v + v } -> std::convertible_to<Tv>;
  };

  // Forward declaration so the trait below can specialize on Tensor<X>
  // without pulling in tensor.hpp here.
  template <typename T>
  class Tensor;

  /// Trait to explicitly disable `LinearTag` for value types that
  /// technically satisfy `LinearlyBlendable` but where pointwise blending
  /// is semantically invalid (e.g., Tensor<X> — blending two tensors with
  /// different shapes is undefined). `TimeSeries::set_interpolation(Linear)`
  /// consults this and throws when it's true.
  template <typename Tv>
  struct disables_linear_interpolation : std::false_type {};

  template <typename X>
  struct disables_linear_interpolation<Tensor<X>> : std::true_type {};

  template <typename Tv>
  inline constexpr bool disables_linear_interpolation_v =
      disables_linear_interpolation<Tv>::value;

  /// Extension point for user-defined interpolation (e.g., optimal transport
  /// for images). Subclass and override `evaluate`; wrap in `CustomStrategy`
  /// and attach via `TimeSeries::set_strategy`.
  ///
  /// Called once per query batch. `TimeSeries::evaluate_batch` pre-computes
  /// the bracketing breakpoints for each query and passes five parallel
  /// vectors of length n_queries. At the right domain boundary,
  /// `t_rights == t_lefts` and `v_rights == v_lefts`; return `v_lefts[i]`
  /// (or another sensible fallback) in that case.
  template <typename Tt, typename Tv>
  class InterpolationStrategy
  {
  public:
    virtual ~InterpolationStrategy() = default;

    virtual std::vector<Tv> evaluate(
        const std::vector<Tt>& queries,
        const std::vector<Tt>& t_lefts,
        const std::vector<Tt>& t_rights,
        const std::vector<Tv>& v_lefts,
        const std::vector<Tv>& v_rights) const = 0;
  };

  /// Variant alternatives for `InterpolationChoice`. `NearestTag` and
  /// `LinearTag` are stateless — the corresponding interpolation logic
  /// lives inline in `TimeSeries::evaluate_batch`. `CustomStrategy` holds a
  /// shared_ptr so a single user strategy instance can be shared across
  /// many `TimeSeries` (e.g., one OT model applied to many series).
  struct NearestTag {};
  struct LinearTag {};

  template <typename Tt, typename Tv>
  struct CustomStrategy
  {
    std::shared_ptr<InterpolationStrategy<Tt, Tv>> ptr;
  };

  template <typename Tt, typename Tv>
  using InterpolationChoice =
      std::variant<NearestTag, LinearTag, CustomStrategy<Tt, Tv>>;

  /// True if the variant currently holds a `CustomStrategy`.
  template <typename Tt, typename Tv>
  [[nodiscard]] inline bool holds_custom_strategy(
      const InterpolationChoice<Tt, Tv>& choice) noexcept
  {
    return std::holds_alternative<CustomStrategy<Tt, Tv>>(choice);
  }

  /// Map the variant to the serialization enum. Only meaningful when the
  /// variant does not hold a `CustomStrategy`; callers should check first.
  template <typename Tt, typename Tv>
  [[nodiscard]] inline InterpolationMode to_mode(
      const InterpolationChoice<Tt, Tv>& choice) noexcept
  {
    if (std::holds_alternative<LinearTag>(choice))
      return InterpolationMode::Linear;
    return InterpolationMode::Nearest;
  }
}

#endif // MPCF_TIMESERIES_INTERPOLATION_H
