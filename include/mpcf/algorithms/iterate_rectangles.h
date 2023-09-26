#ifndef MPCF_ALGORITHM_ITERATE_RECTANGLES_H
#define MPCF_ALGORITHM_ITERATE_RECTANGLES_H

#include <algorithm>

namespace mpcf
{
  template <typename TPcf, typename FCb>
  void iterate_rectangles(const TPcf& f, const TPcf& g, 
    typename TPcf::time_type a, typename TPcf::time_type b,
    FCb cb)
  {
    using TTime = typename TPcf::time_type;
    using TVal = typename TPcf::value_type;

    TTime t = 0;
    TTime tprev = 0;

    TVal fv = 0;
    TVal gv = 0;

    auto fi = 0; // max_time_prior_to(f, a);
    auto gi = 0; // max_time_prior_to(g, a);

    auto const & fpts = f.points();
    auto const & gpts = g.points();

    auto fsz = fpts.size();
    auto gsz = gpts.size();

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
}

#endif
