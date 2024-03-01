#ifndef MPCF_RECTANGLE_H
#define MPCF_RECTANGLE_H

namespace mpcf
{
  template <typename Tt, typename Tv>
  struct Rectangle
  {
    Tt left = 0;
    Tt right = 0;
    Tv top = 0;
    Tv bottom = 0;

    Rectangle() = default;
    Rectangle(Tt l, Tt r, Tv t, Tv b)
      : left(l), right(r), top(t), bottom(b)
    { }
  };

  template <typename Tt, typename Tv>
  struct Segment
  {
    Tt left = 0;
    Tt right = 0;
    Tv value = 0;

    Segment() = default;
    Segment(Tt l, Tt r, Tv v)
      : left(l), right(r), value(v)
    { }
  };
}

#endif
