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

#ifndef MPCF_PCF_H
#define MPCF_PCF_H

#include "point.h"
#include "rectangle.h"
#include "../algorithms/functional/reduce.h"

#include <vector>
#include <iostream>
#include <ostream>

namespace mpcf
{
  template <typename Tt, typename Tv = Tt>
  class Pcf
  {
  public:
    using point_type = Point<Tt, Tv>;
    using time_type = typename point_type::time_type;
    using value_type = typename point_type::value_type;

    using reduction_functional_type = value_type(time_type, time_type, value_type, value_type);
    using rectangle_type = Rectangle<Tt, Tv>;
    using segment_type = Segment<Tt, Tv>;

    Pcf()
    {
      m_points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(0));
    }

    explicit Pcf(value_type v)
    {
      m_points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(v));
    }

    explicit Pcf(std::vector<Point<Tt, Tv>>&& pts)
      : m_points(std::move(pts))
    {

    }

    Pcf(std::initializer_list<std::pair<Tt, Tv>> pts)
      requires (!FloatType<Tt> || !FloatType<Tv>)
    {
      // We can't have this constructor if we work with floating point types because of the overloads below, but we
      // still want to be able to construct from pairs if we are using non-float types.
      initFromPairs(pts);
    }

    Pcf(std::initializer_list<std::pair<float64_t, float64_t>> pts)
      requires (std::is_same_v<Tt, float64_t> && std::is_same_v<Tv, float64_t>)
    {
      initFromPairs(pts);
    }

    Pcf(std::initializer_list<std::pair<float32_t, float32_t>> pts)
      requires (std::is_same_v<Tt, float32_t> && std::is_same_v<Tv, float32_t>)
    {
      initFromPairs(pts);
    }

    std::string to_string() const
    {
      std::stringstream ss;
      for (auto i = 0ul; i < m_points.size(); ++i)
      {
        if (i != 0ul)
        {
          ss << ", ";
        }
        ss << "f(" << m_points[i].t << ") = " << m_points[i].v;
      }
      return ss.str();
    }
    
    void debug_print() const
    {
      std::cout << to_string() << std::endl;
    }

    Pcf& operator+=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.top + rect.bottom;
        });
      return *this;
    }

    Pcf& operator-=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.top - rect.bottom;
        });
      return *this;
    }

    Pcf& operator*=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.top * rect.bottom;
        });
      return *this;
    }

    Pcf& operator/=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.top / rect.bottom;
        });
      return *this;
    }

    template <typename T>
    Pcf& operator/=(T v)
    {
      static_assert(std::is_arithmetic<T>::value, "Division by non-arithmetic type");
      for (auto & pt : m_points)
      {
        pt.v /= v;
      }
      return *this;
    }

    template <typename T>
    Pcf& operator*=(T v)
    {
      static_assert(std::is_arithmetic<T>::value, "Multiplication by non-arithmetic type");
      for (auto & pt : m_points)
      {
        pt.v *= v;
      }
      return *this;
    }
    
    bool operator==(const Pcf& rhs) const
    {
      return m_points == rhs.m_points;
    }
    
    bool operator!=(const Pcf& rhs) const
    {
      return m_points != rhs.m_points;
    }

    [[nodiscard]] value_type evaluate(time_type t) const
    {
      auto it = std::upper_bound(m_points.begin(), m_points.end(), t,
        [](time_type time, const point_type& pt) { return time < pt.t; });

      if (it == m_points.begin())
      {
        throw std::domain_error("Cannot evaluate PCF before time 0.");
      }

      --it;
      return it->v;
    }

    template <typename TIn, typename TOut>
    void evaluate(const TIn& sorted_times, TOut& out, size_t n) const
    {
      auto it = m_points.begin();
      auto end = m_points.end();

      for (size_t i = 0; i < n; ++i)
      {
        auto t = sorted_times(i);

        if (t < m_points.front().t)
        {
          throw std::domain_error("Cannot evaluate PCF before time 0.");
        }

        while (std::next(it) != end && std::next(it)->t <= t)
        {
          ++it;
        }

        out(i) = it->v;
      }
    }

    [[nodiscard]] const std::vector<point_type>& points() const { return m_points; }
    [[nodiscard]] size_t size() const noexcept { return m_points.size(); }
    
    void swap(Pcf& other) noexcept
    {
      m_points.swap(other.m_points);
    }
    
    template <typename Ut, typename Uv>
    friend void PrintTo(const Pcf& f, std::ostream* os);
    

  private:
    template <typename T1, typename T2>
    void initFromPairs(std::initializer_list<std::pair<T1, T2>> pts)
    {
      m_points.reserve(pts.size());
      for (auto const& pt : pts)
      {
        m_points.emplace_back(static_cast<Tt>(pt.first), static_cast<Tv>(pt.second));
      }
    }

    std::vector<point_type> m_points;
  };
  
  using Pcf_f32 = Pcf<float32_t, float32_t>;
  using Pcf_f64 = Pcf<float64_t, float64_t>;
  
  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator+(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.top + rect.bottom;
    });
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator-(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    Pcf<Tt, Tv> ret = f;
    ret -= g;
    return ret;
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator*(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.top * rect.bottom;
    });
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator/(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.top / rect.bottom;
    });
  }

  template <typename Tt, typename Tv, typename Tdiv>
  requires std::is_arithmetic_v<Tdiv>
  [[nodiscard]] Pcf<Tt, Tv> operator/(const Pcf<Tt, Tv>& f, Tdiv val)
  {
    Pcf<Tt, Tv> ret = f;
    ret /= val;
    return ret;
  }

  template <typename Tt, typename Tv, typename Ts>
  requires std::is_arithmetic_v<Ts>
  [[nodiscard]] Pcf<Tt, Tv> operator*(const Pcf<Tt, Tv>& f, Ts val)
  {
    Pcf<Tt, Tv> ret = f;
    ret *= val;
    return ret;
  }

  template <typename Tt, typename Tv, typename Ts>
  requires std::is_arithmetic_v<Ts>
  [[nodiscard]] Pcf<Tt, Tv> operator*(Ts val, const Pcf<Tt, Tv>& f)
  {
    Pcf<Tt, Tv> ret = f;
    ret *= val;
    return ret;
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> average(const std::vector<Pcf<Tt, Tv>>& fs, size_t chunksz = 2ul)
  {
    auto f = parallel_reduce(fs.begin(), fs.end(), [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
      return rect.top + rect.bottom;
    }, chunksz);
    return f / static_cast<Tv>(fs.size());
  }

  template <typename Tt, typename Tv>
  void PrintTo(const Pcf<Tt, Tv>& f, std::ostream* os)
  {
    *os << f.to_string();
  }

  template <typename F>
  concept PcfLike = requires(F f, const F cf)
  {
    typename F::point_type;
    typename F::time_type;
    typename F::value_type;
    typename F::rectangle_type;
    typename F::segment_type;

    requires PointLike<typename F::point_type>;

    { cf.points() } -> std::convertible_to<const std::vector<typename F::point_type>&>;
    { cf.size()   } -> std::convertible_to<std::size_t>;

    { cf == cf } -> std::convertible_to<bool>;
    { cf != cf } -> std::convertible_to<bool>;
  };
}

#endif
