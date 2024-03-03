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

    exec.cpu()->run(std::move(flow)).wait();
  }

  template <typename PcfT>
  inline typename PcfT::value_type l1_norm(const PcfT& f)
  {
    typename PcfT::value_type out{ 0 };

    iterate_segments(f.points().cbegin(), f.points().cend(), typename PcfT::time_type{ 0 }, infinite_time<PcfT>(), [&out](typename PcfT::segment_type& seg) {
      out += (seg.right - seg.left) * std::abs(seg.value);
      });

    return out;
  }

  template <typename PcfT>
  inline typename PcfT::value_type l2_norm(const PcfT& f)
  {
    typename PcfT::value_type out{ 0 };

    iterate_segments(f.points().cbegin(), f.points().cend(), typename PcfT::time_type{ 0 }, infinite_time<PcfT>(), [&out](typename PcfT::segment_type& seg) {
      out += (seg.right - seg.left) * seg.value * seg.value;
      });

    return std::sqrt(out);
  }

  template <typename PcfT>
  inline typename PcfT::value_type lp_norm(const PcfT& f, typename PcfT::value_type p)
  {
    typename PcfT::value_type out{ 0 };

    iterate_segments(f.points().cbegin(), f.points().cend(), typename PcfT::time_type{ 0 }, infinite_time<PcfT>(), [&out, p](typename PcfT::segment_type& seg) {
      out += (seg.right - seg.left) * std::pow(std::abs(seg.value), p);
      });

    return std::pow(out, typename PcfT::value_type{1} / p);
  }

  template <typename PcfT>
  inline typename PcfT::value_type linfinity_norm(const PcfT& f)
  {
    typename PcfT::value_type out{ 0 }; // No PCF should be empty, but if that happens we adopt the convention that L_inf=0
    for (auto const & pt : f.points())
    {
      out = std::max(out, std::abs(pt.v));
    }
    return out;
  }

}

#endif
