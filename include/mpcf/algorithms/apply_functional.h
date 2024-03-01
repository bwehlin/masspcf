#ifndef MPCF_ALGORITHM_APPLY_FUNCTIONAL_H
#define MPCF_ALGORITHM_APPLY_FUNCTIONAL_H

#include "../pcf.h"
#include "../executor.h"

#include <vector>
#include <functional>
#include <iterator>
#include <limits>

#include <taskflow/algorithm/transform.hpp>

namespace mpcf
{


  template <typename ForwardPcfIt, typename OutputIt, typename F>
  inline void apply_functional(ForwardPcfIt begin, ForwardPcfIt end, OutputIt beginOut, F functional, mpcf::Executor& exec = mpcf::default_executor())
  {
    using PcfT = decltype(*begin);

    tf::Taskflow flow;
    flow.transform(begin, end, beginOut, [&functional](const PcfT& pcf) {
      return functional(pcf);
      });

    exec.cpu()->run(std::move(flow));
  }

  //iterate_segments(PointFwdIterator beginPoints, PointFwdIterator endPoints, typename decltype(*beginPoints)::time_type a, typename decltype(*beginPoints)::time_type b, FCb cb)
  template <typename PcfT>
  inline typename PcfT::value_type l1_norm(const PcfT& f)
  {
    typename PcfT::value_type out{ 0 };
    iterate_segments(f.points().cbegin(), f.points().cend(), 0.f, 0.f, [&out](typename PcfT::segment_type& seg) {
      out += (seg.right - seg.left) * std::abs(seg.value);
      });
    iterate_segments(f.points().cbegin(), f.points().cend(), typename PcfT::time_type{ 0 }, infinite_time<PcfT>(), [&out](typename PcfT::segment_type& seg) {
      out += (seg.right - seg.left) * std::abs(seg.value);
      });

    return out;
  }
#if 0
  template <typename PcfT>
  inline typename PcfT::value_type l2_norm(const PcfT& f)
  {
    if (f.points().empty())
    {
      return typename PcfT::value_type{ 0 };
    }

    if (f.points().back().v != 0)
    {
      return infinite_value<PcfT>();
    }

    typename PcfT::time_type tLast{ 0 };
    typename PcfT::value_type vLast{ 0 };
    typename PcfT::value_type out{ 0 };

    for (auto const& pt : f.points())
    {
      out += (pt.t - tLast) * std::abs(vLast);
      tLast = pt.t;
      vLast = pt.v;
    }

    return out;
  }

  template <typename PcfT>
  inline typename PcfT::value_type linfinity_norm(const PcfT& f)
  {
    
  }

  template <typename PcfT>
  inline typename PcfT::value_type lp_norm(const PcfT& f, typename PcfT::value_type p)
  {
    if (p == infinite_value<PcfT>())
    {
      return linfinity_norm(f);
    }

    if (f.points().back().v != 0)
    {
      return infinite_value<PcfT>();
    }

    typename PcfT::time_type tLast{ 0 };
    typename PcfT::value_type val{ 0 };
    for (auto const& pt : f.points())
    {
      val += (pt.t - tLast) * std::pow(pt.v, p);
      tLast = pt.t;
    }

    return std::pow(val, PcfT::value_type{ 1 } / p);
  }

#endif

}

#endif
