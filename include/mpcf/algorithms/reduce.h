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

    Accumulator(TOp<TPcf> op, size_t sizeHint)
      : m_op(op)
    { 
      m_pts.reserve(sizeHint);
      m_pts_temp.reserve(sizeHint);
    }

    Accumulator(TOp<TPcf> op, const TPcf& f)
      : m_op(op)
    { 
      m_pts = f.points();
    }

    Accumulator& operator+=(const TPcf& f)
    {
      if (m_pts.empty())
      {
        m_pts = f.points();
        return *this;
      }
      combine_with_(f.points());
      return *this;
    }

    Accumulator& operator+=(const Accumulator& other)
    {
      combine_with_(other.m_pts);
      return *this;
    }

    Accumulator& operator+=(Accumulator&& other)
    {
      combine_with_(other.m_pts);
      return *this;
    }

    Accumulator(Accumulator&& other)
      : m_op(other.m_op)
      , m_pts(std::move(other.m_pts))
      , m_pts_temp(std::move(other.m_pts_temp))
    { }

    Accumulator(const Accumulator& other)
      : m_op(other.m_op)
      , m_pts(other.m_pts)
      , m_pts_temp(other.m_pts_temp)
    { }

    Accumulator& operator=(Accumulator&& other)
    {
      if (this == &other)
      {
        return *this;
      }
      m_op = other.m_op;
      m_pts = std::move(other.m_pts);
      m_pts_temp = std::move(other.m_pts_temp);
      return *this;
    }

    Accumulator& operator=(const Accumulator& other)
    {
      if (this == &other)
      {
        return *this;
      }
      m_op = other.m_op;
      m_pts = other.m_pts;
      m_pts_temp = other.m_pts_temp;
      return *this;
    }

    void combine_with_(const std::vector<point_type>& other)
    {
      using rectangle_type = typename TPcf::rectangle_type;
      auto i = 0ul;
      m_pts_temp.clear();
      m_pts_temp.resize(m_pts.size() + other.size() + 1);

      iterate_rectangles(m_pts, other, 0, 1000, [&i, this](const rectangle_type& rect){
        ++i;
        m_pts_temp[i].t = rect.left;
        m_pts_temp[i].v = m_op(rect);
      });

      m_pts_temp.resize(i + 1);
      m_pts_temp.swap(m_pts);
    }

    TOp<TPcf> m_op;
    std::vector<point_type> m_pts;
    std::vector<point_type> m_pts_temp;
  };

  struct TaskWithTag
  {
    tf::Task task;
    size_t tag;
  };

  template <typename TPcf>
  TPcf mem_parallel_reduce(const std::vector<TPcf>& fs, TOp<TPcf> op, size_t chunkszFirst = 8)
  {
    auto chunksz = 2;
    auto blocks = subdivide(chunkszFirst , fs.size());

    auto nAll = std::accumulate(fs.begin(), fs.end(), 0ul, [](size_t n, const TPcf& f){ return f.points().size() + n; });

    std::vector<Accumulator<TPcf>> accumulators;
    accumulators.resize(blocks.size(), Accumulator<TPcf>(op, nAll));

    tf::Taskflow taskflow;
    tf::Executor exec;

    std::list<std::vector<TaskWithTag>> taskLevels;
    auto & topLevelTasks = taskLevels.emplace_back(accumulators.size());
    for (auto iBlock = 0ul; iBlock < blocks.size(); ++iBlock)
    {
      auto const & block = blocks[iBlock];
      topLevelTasks[iBlock].tag = iBlock; // Accumulator id
      topLevelTasks[iBlock].task = taskflow.emplace([iBlock, block, &accumulators, &fs](){
        for (auto i = block.first; i <= block.second; ++i)
        {
          accumulators[iBlock] += fs[i];
        }
      }); //.name("f" + std::to_string(block.first) + "..." + std::to_string(block.second) + " -> acc" + std::to_string(iBlock));
    }

    std::vector<size_t> tags;
    while (taskLevels.back().size() > 1ul)
    {
      auto const & prevLevel = taskLevels.back();
      blocks = subdivide(chunksz, prevLevel.size());
      
      //std::vector<TaskWithTag> tasks(blocks.size());
      auto & tasks = taskLevels.emplace_back(blocks.size());

      for (auto iBlock = 0ul; iBlock < blocks.size(); ++iBlock)
      {
        auto const & block = blocks[iBlock];
        auto targetAcc = prevLevel[block.first].tag;
        tasks[iBlock].tag = targetAcc;
#if 0
        std::string srcAccs;
        for (auto i = block.first + 1; i <= block.second; ++i)
        {
          auto srcAcc = prevLevel[i].tag;
          if (i != block.first + 1)
          {
            srcAccs += ",";
          }
          srcAccs += std::to_string(srcAcc);
        }
#endif
        tasks[iBlock].task = taskflow.emplace([targetAcc, block, &accumulators, &prevLevel](){
          for (auto i = block.first + 1; i <= block.second; ++i)
          {
            auto srcAcc = prevLevel[i].tag;
            accumulators[targetAcc] += accumulators[srcAcc];
          }
        }); //.name(srcAccs + "->" + std::to_string(targetAcc));
        for (auto iPrev = block.first; iPrev <= block.second; ++iPrev)
        {
          tasks[iBlock].task.succeed(prevLevel[iPrev].task);
        }
      }

    }


    //taskflow.dump(std::cout);
    exec.run(taskflow).wait();

    return TPcf(std::move(accumulators[0].m_pts));

#if 0

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
      return std::move(f += g);
    });
    exec.run(taskflow).wait();
    return TPcf(std::move(ret.m_pts));

#endif

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
