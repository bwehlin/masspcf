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

#ifndef MPCF_POINT_H
#define MPCF_POINT_H

#include <limits>

namespace mpcf
{
  template <typename T>
  constexpr T infinity()
  {
    if constexpr (std::numeric_limits<T>::has_infinity)
    {
      return std::numeric_limits<T>::infinity();
    }
    else
    {
      return (std::numeric_limits<T>::max)();
    }
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

    Tt t = 0.; // time
    Tv v = 0.; // value

    Point() = default;
    Point(Tt it, Tv iv)
      : t(it), v(iv) { }
    
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
  
  using Point_f32 = Point<float, float>;
  using Point_f64 = Point<double, double>;
}

#endif
