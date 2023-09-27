#ifndef MPCF_POINT_H
#define MPCF_POINT_H

namespace mpcf
{
  template <typename Tt, typename Tv>
  struct Point
  {
    using time_type = Tt;
    using value_type = Tv;

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
