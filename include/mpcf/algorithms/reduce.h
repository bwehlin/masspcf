#ifndef MPCF_ALGORITHM_REDUCE_H
#define MPCF_ALGORITHM_REDUCE_H

#include "iterate_rectangles.h"
#include "subdivide.h"

#include <functional>
#include <vector>
#include <numeric>
#include <algorithm>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace mpcf
{
  template <typename TPcf>
  using TOp = std::function<typename TPcf::value_type(const typename TPcf::rectangle_type&)>;

  template <typename TPcf>
  TPcf combine(const TPcf& f, const TPcf& g, TOp<TPcf> op)
  {
    using point_type = typename TPcf::point_type;
    using rectangle_type = typename TPcf::rectangle_type;

    std::vector<point_type> retPts;
    std::size_t npts = 0;
    iterate_rectangles(f, g, 0, 1000, [&npts](const rectangle_type&){ ++npts; });
    retPts.resize(npts);
    auto i = 0ul;
    iterate_rectangles(f, g, 0, 1000, [&retPts, &i, &op](const rectangle_type& rect){
      retPts[i++] = point_type(rect.left, op(rect));
    });

    return TPcf(std::move(retPts));
  }

  template <typename TPcf>
  TPcf reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    return std::reduce(fs.begin(), fs.end(), TPcf(), [&op](const TPcf& f, const TPcf& g) {
      return combine(f, g, op);
    });
  }
  
  template <typename TPcf>
  TPcf parallel_reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    tf::Taskflow taskflow;
    tf::Executor exec;
    TPcf f;
    /*auto task =*/ taskflow.reduce(fs.begin(), fs.end(), f, [&op](const TPcf& f, const TPcf& g) {
      return combine(f, g, op);
    });
    exec.run(taskflow).wait();
    return f;
  }

  template <typename TPcf>
  class Accumulator
  {
  public:
    using point_type = typename TPcf::point_type;

    Accumulator(TOp<TPcf> op)
      : m_op(op)
    { 
      m_pts.emplace_back(0, 0);
    }

    Accumulator(TOp<TPcf> op, const TPcf& f)
      : m_op(op)
    { 
      m_pts = f.points();
    }

    Accumulator& operator+=(const TPcf& f)
    {
      combine_with_(f.points());
      return *this;
    }

    Accumulator& operator+=(const Accumulator& other)
    {
      combine_with_(other.m_pts);
      return *this;
    }

    void combine_with_(const std::vector<point_type>& other)
    {
      using rectangle_type = typename TPcf::rectangle_type;

      //std::size_t npts = 0;
      //iterate_rectangles(m_pts, other, 0, 1000, [&npts](const rectangle_type&){ ++npts; });
      //m_pts_temp.resize(npts);
      auto i = 0ul;
      m_pts_temp.clear();
      m_pts_temp.resize(m_pts.size() + other.size() + 1);

      iterate_rectangles(m_pts, other, 0, 1000, [&i, this](const rectangle_type& rect){
        //m_pts_temp[i++] = point_type(rect.left, m_op(rect));
        ++i;
        m_pts_temp[i].t = rect.left;
        m_pts_temp[i].v = m_op(rect); // = point_type(rect.left, m_op(rect));
      });

      m_pts_temp.resize(i + 1);

      m_pts_temp.swap(m_pts);
    }

    TOp<TPcf> m_op;
    std::vector<point_type> m_pts;
    std::vector<point_type> m_pts_temp;
  };

  template <typename TPcf>
  TPcf mem_parallel_reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    auto boundaries = subdivide(256, fs.size());

    std::vector<Accumulator<TPcf>> accumulators;
    accumulators.resize(boundaries.size(), Accumulator<TPcf>(op));

    tf::Taskflow taskflow;
    tf::Executor exec;

    taskflow.for_each_index(0ul, boundaries.size(), 1ul, [&fs, &accumulators, &boundaries](size_t iBlock) {
      auto const & block = boundaries[iBlock];
      for (auto i = block.first; i <= block.second; ++i)
      {
        accumulators[iBlock] += fs[i];
      }
    });
    exec.run(taskflow).wait();
    for (auto it = accumulators.begin() + 1; it != accumulators.end(); ++it)
    {
      accumulators[0] += *it;
    }
    return TPcf(std::move(accumulators[0].m_pts));

    //std::vector<Accumulator<TPcf>> accumulators;
    accumulators.reserve(fs.size());
    for (auto const & f : fs)
    {
      accumulators.emplace_back(op, f);
    }

    //tf::Taskflow taskflow;
    //tf::Executor exec;
    Accumulator<TPcf> ret(op);
    /*auto task =*/ taskflow.reduce(accumulators.begin(), accumulators.end(), ret, 
    [&op](Accumulator<TPcf>& f, Accumulator<TPcf>& g) {
      return f += g;
    });
    exec.run(taskflow).wait();
    return TPcf(std::move(ret.m_pts));

#if 0
    Accumulator<TPcf> acc(op);
    for (auto const & f : fs)
    {
      acc += f;
    }
#endif

    //auto acc = std::reduce(fs.begin(), fs.end(), Accumulator<TPcf>(op), [](Accumulator<TPcf>& lhs, const Accumulator<TPcf>& rhs) {
     // return lhs += rhs;
    //});
    //return TPcf(std::move(acc.m_pts));
    return fs[0];
  }
}

#endif
