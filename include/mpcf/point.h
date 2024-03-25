#ifndef MPCF_POINT_H
#define MPCF_POINT_H

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
  };
  
  using Point_f32 = Point<float, float>;
  using Point_f64 = Point<double, double>;
}

#endif
