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

#ifndef MPCF_POINT_H
#define MPCF_POINT_H

#include "../config.hpp"

#include <limits>

namespace mpcf
{
  template <typename T>
  constexpr T infinity()
  {
    return (std::numeric_limits<T>::max)();
  }
  
  template <typename Tt, typename Tv>
  struct Point
  {
    using time_type = Tt;
    using value_type = Tv;
    
    constexpr static time_type zero_time()
    {
      return Tt(0);
    }
    
    constexpr static time_type infinite_time()
    {
      return infinity<time_type>();
    }

    Tt t = static_cast<Tt>(0.); // time
    Tv v = static_cast<Tv>(0.); // value

    Point() = default;
    Point(Tt it, Tv iv)
      : t(it), v(iv) { }

    template <typename T1, typename T2>
    Point(T1 it, T2 iv)
      : t(static_cast<Tt>(it)), v(static_cast<Tv>(iv)) { }

    
    bool operator==(const Point& rhs) const
    {
      return t == rhs.t && v == rhs.v;
    }
    
    bool operator!=(const Point& rhs) const
    {
      return t != rhs.t || v != rhs.v;
    }
  };

  template <typename Tt, typename Tv>
  struct OrderByTimeAscending
  {
    [[nodiscard]] bool operator()(const mpcf::Point<Tt, Tv>& a, const mpcf::Point<Tt, Tv>& b) const noexcept
    {
      return a.t < b.t;
    }
  };
  
  using Point_f32 = Point<float32_t, float32_t>;
  using Point_f64 = Point<float64_t, float64_t>;
  using Point_i32 = Point<int32_t, int32_t>;
  using Point_i64 = Point<int64_t, int64_t>;

  template <typename P>
  concept PointLike = requires(P p)
  {
    typename P::time_type;
    typename P::value_type;

    { p.t } -> std::convertible_to<typename P::time_type>;
    { p.v } -> std::convertible_to<typename P::value_type>;

    { p == p } -> std::convertible_to<bool>;
    { p != p } -> std::convertible_to<bool>;
  };
}

#endif
