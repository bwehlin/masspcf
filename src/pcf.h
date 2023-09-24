#ifndef MPCF_POINT_H
#define MPCF_POINT_H

#include "point.h"
#include "rectangle.h"
#include "algorithms/reduce.h"

#include <vector>
#include <iostream>

namespace mpcf
{
  template <typename Tt, typename Tv>
  class Pcf
  {
  public:
    using point_type = Point<Tt, Tv>;
    using time_type = typename point_type::time_type;
    using value_type = typename point_type::value_type;

    using reduction_functional_type = value_type(time_type, time_type, value_type, value_type);
    using rectangle_type = Rectangle<Tt, Tv>;

    Pcf()
    {
      m_points.emplace_back(0, 0);
    }

    Pcf(std::vector<Point<Tt, Tv>>&& pts)
      : m_points(std::move(pts))
    {

    }

    void debugPrint() const
    {
      for (auto i = 0ul; i < m_points.size(); ++i)
      {
        if (i != 0ul)
        {
          std::cout << ", ";
        }
        std::cout << "f(" << m_points[i].t << ") = " << m_points[i].v;
      }
      std::cout << std::endl;
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

    const std::vector<point_type>& points() const { return m_points; }

  private:
    std::vector<point_type> m_points;
  };

  template <typename Tt, typename Tv>
  Pcf<Tt, Tv> operator+(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.top + rect.bottom;
    });
  }

  template <typename Tt, typename Tv>
  Pcf<Tt, Tv> operator-(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.top - rect.bottom;
    });
  }

  template <typename Tt, typename Tv, typename Tdiv>
  Pcf<Tt, Tv> operator/(const Pcf<Tt, Tv>& f, Tdiv val)
  {
    Pcf<Tt, Tv> ret = f;
    ret /= val;
    return ret;
  }

  template <typename Tt, typename Tv>
  Pcf<Tt, Tv> average(const std::vector<Pcf<Tt, Tv>>& fs)
  {
    auto f = reduce(fs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.top + rect.bottom;
    });
    return f / static_cast<Tv>(fs.size());
  }
  
}

#endif
