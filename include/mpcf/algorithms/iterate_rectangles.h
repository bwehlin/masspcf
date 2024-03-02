#ifndef MPCF_ALGORITHM_ITERATE_RECTANGLES_H
#define MPCF_ALGORITHM_ITERATE_RECTANGLES_H

#include <algorithm>
#include <vector>
#include "../rectangle.h"

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

  template <typename PcfT>
  constexpr typename PcfT::value_type infinite_value()
  {
    return infinity<typename PcfT::value_type>();
  }

  template <typename PcfT>
  constexpr typename PcfT::time_type infinite_time()
  {
    return infinity<typename PcfT::time_type>();
  }

  template <typename Point, typename FCb>
  void iterate_rectangles(const std::vector<Point>& fpts, const std::vector<Point>& gpts, 
    typename Point::time_type /*a*/, typename Point::time_type b,
    FCb cb)
  {
    using TTime = typename Point::time_type;
    using TVal = typename Point::value_type;

    TTime t = 0;
    TTime tprev = 0;

    TVal fv = 0;
    TVal gv = 0;

    size_t fi = 0; // max_time_prior_to(f, a);
    size_t gi = 0; // max_time_prior_to(g, a);

    size_t fsz = fpts.size();
    size_t gsz = gpts.size();

    Rectangle<TTime, TVal> rect;

    while (t < b)
    {
      tprev = t;
      fv = fpts[fi].v;
      gv = gpts[gi].v;

      if (fi + 1 < fsz && gi + 1 < gsz)
      {
        auto delta = fpts[fi+1].t - gpts[gi+1].t;
        if (delta <= 0)
        {
          ++fi;
        }
        if (delta >= 0)
        {
          ++gi;
        }
      }
      else
      {
        if (fi + 1 < fsz)
        {
          ++fi;
        }
        else if (gi + 1 < gsz)
        {
          ++gi;
        }
        else
        {
          rect.left = tprev;
          rect.right = b;
          rect.top = fv;
          rect.bottom = gv;

          cb(rect);

          return;
        }
      }

      t = std::max(fpts[fi].t, gpts[gi].t);

      rect.left = tprev;
      rect.right = t;
      rect.top = fv;
      rect.bottom = gv;

      cb(rect);
    }
  }

  template <typename TPcf, typename FCb>
  void iterate_rectangles(const TPcf& f, const TPcf& g, 
    typename TPcf::time_type a, typename TPcf::time_type b,
    FCb cb)
  {
    auto const & fpts = f.points();
    auto const & gpts = g.points();
    iterate_rectangles(fpts, gpts, a, b, cb);
  }

  template <typename PointFwdIterator, typename FCb>
  inline void iterate_segments(PointFwdIterator beginPoints, PointFwdIterator endPoints, 
    typename std::iterator_traits<PointFwdIterator>::value_type::time_type a, 
    typename std::iterator_traits<PointFwdIterator>::value_type::time_type b, 
    FCb cb)
  {
    using TPoint = typename std::iterator_traits<PointFwdIterator>::value_type;
    using TTime = typename TPoint::time_type;
    using TVal = typename TPoint::value_type;

    if (beginPoints == endPoints)
    {
      return;
    }

    // TODO: start at 'a', end at 'b'

    TTime tprev = beginPoints->t;
    TVal vprev = beginPoints->v;

    Segment<TTime, TVal> seg;
    for (auto it = std::next(beginPoints); it != endPoints; ++it)
    {
      seg.left = tprev;
      seg.right = it->t;
      seg.value = vprev;

      cb(seg);

      tprev = seg.right;
      vprev = it->v;
    }

    seg.left = tprev;
    seg.right = (std::numeric_limits<TTime>::max)(); // float (0 * inf) does not produce 0, so use max instead.
    seg.value = vprev;

    cb(seg);
    
  }
}

#endif
