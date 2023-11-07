#ifndef MPCF_ALGORITHM_REDUCE_H
#define MPCF_ALGORITHM_REDUCE_H

#include "iterate_rectangles.h"
#include "subdivide.h"
#include "../executor.h"

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
  class Accumulator
  {
  public:
    using point_type = typename TPcf::point_type;
    using time_type = typename TPcf::time_type;

    Accumulator(TOp<TPcf> op, size_t sizeHint = 0ul)
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
      if (m_pts.empty())
      {
        m_pts = other.m_pts;
        return *this;
      }
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
      if (other.empty())
      {
        return;
      }

      if (m_pts.empty())
      {
        m_pts.emplace_back(0, 0);
      }

      using rectangle_type = typename TPcf::rectangle_type;
      auto i = 0ul;
      m_pts_temp.clear();
      m_pts_temp.resize(m_pts.size() + other.size() + 1);

      auto old_val = m_pts.front().v;

      iterate_rectangles(m_pts, other, 0, std::numeric_limits<time_type>::max(), [&i, &old_val, this](const rectangle_type& rect) {

        auto new_val = m_op(rect);
        if (new_val != old_val)
        {
          ++i;
          m_pts_temp[i].t = rect.left;
          m_pts_temp[i].v = m_op(rect);
          old_val = new_val;
        }
      });

      m_pts_temp.resize(i + 1);
      m_pts_temp.swap(m_pts);
    }

    TOp<TPcf> m_op;
    std::vector<point_type> m_pts;
    std::vector<point_type> m_pts_temp;
  };

  template <typename TPcf>
  Accumulator<TPcf> operator+(const Accumulator<TPcf>& lhs, const Accumulator<TPcf>& rhs)
  {
    Accumulator<TPcf> ret(lhs);
    ret += rhs;
    return ret;
  }

  struct TaskWithTag
  {
    tf::Task task;
    size_t tag;
  };

  template <typename T>
  class ThreadLocalStorage
  {
  public:
    ThreadLocalStorage(Executor& exec, const T& init = {})
      : m_data(exec.cpu()->num_workers(), init)
      , m_exec(exec)
    { }

    T& local()
    {
      return m_data[m_exec.cpu()->this_worker_id()];
    }

    tf::Task parallel_reduce(tf::Taskflow& flow, T& result)
    {
      return flow.reduce(m_data.begin(), m_data.end(), result, [](const T& lhs, const T& rhs) { return lhs + rhs; });
    }

  private:
    std::vector<T> m_data;
    Executor& m_exec;
  };

  template <typename TPcf>
  TPcf parallel_reduce_2(const std::vector<TPcf>& fs, TOp<TPcf> op, Executor& exec)
  {
    auto nAll = std::accumulate(fs.begin(), fs.end(), static_cast<size_t>(0ul), [](size_t n, const TPcf& f) { return f.points().size() + n; });

    ThreadLocalStorage<Accumulator<TPcf>> accumulators(exec, Accumulator<TPcf>(op, nAll));
    tf::Taskflow flow;
    std::vector<tf::Task> tasks;
    Accumulator<TPcf> retAcc(op, nAll);

    tasks.emplace_back(flow.for_each(fs.begin(), fs.end(), [&accumulators](const TPcf& f) {
      accumulators.local() += f;
      }));

    tasks.emplace_back(accumulators.parallel_reduce(flow, retAcc));

    flow.linearize(tasks);
    
    exec.cpu()->run(std::move(flow)).wait();

    return TPcf(std::move(retAcc.m_pts));
  }
  
  template <typename TPcf>
  TPcf parallel_reduce(const std::vector<TPcf>& fs, TOp<TPcf> op, size_t chunkszFirst = 8)
  {
    auto chunksz = 2;
    auto blocks = subdivide(chunkszFirst , fs.size());

    auto nAll = std::accumulate(fs.begin(), fs.end(), static_cast<size_t>(0ul), [](size_t n, const TPcf& f){ return f.points().size() + n; });

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
      });
    }

    std::vector<size_t> tags;
    while (taskLevels.back().size() > 1ul)
    {
      auto const & prevLevel = taskLevels.back();
      blocks = subdivide(chunksz, prevLevel.size());
      
      auto & tasks = taskLevels.emplace_back(blocks.size());

      for (auto iBlock = 0ul; iBlock < blocks.size(); ++iBlock)
      {
        auto const & block = blocks[iBlock];
        auto targetAcc = prevLevel[block.first].tag;
        tasks[iBlock].tag = targetAcc;
        tasks[iBlock].task = taskflow.emplace([targetAcc, block, &accumulators, &prevLevel](){
          for (auto i = block.first + 1; i <= block.second; ++i)
          {
            auto srcAcc = prevLevel[i].tag;
            accumulators[targetAcc] += accumulators[srcAcc];
          }
        });
        for (auto iPrev = block.first; iPrev <= block.second; ++iPrev)
        {
          tasks[iBlock].task.succeed(prevLevel[iPrev].task);
        }
      }

    }

    exec.run(taskflow).wait();

    return TPcf(std::move(accumulators[0].m_pts));
  }
}

#endif
