#ifndef MPCF_TASK_H
#define MPCF_TASK_H

#include <atomic>
#include <functional>
#include <iostream>

#include <taskflow/taskflow.hpp>

#include "executor.h"

namespace mpcf
{
  template <typename RetT>
  class StoppableTask
  {
  public:
    void request_stop()
    {
      m_stop_requested.store(true);
      // m_future.cancel(); // Some issue with this at the moment.
    }
    
    bool stop_requested() const
    {
      return m_stop_requested.load();
    }
    
    const tf::Future<RetT>& future() const
    {
      return m_future;
    }
    
    tf::Future<RetT>& future()
    {
      return m_future;
    }
    
    StoppableTask& start_async(mpcf::Executor& exec)
    {
      m_stop_requested.store(false);
      m_work_completed.store(0ul);
      m_future = run_async(exec);
      return *this;
    }
    
    size_t work_completed() const
    {
      return m_work_completed.load();
    }
    
    size_t work_total() const
    {
      return m_work_total;
    }

    size_t work_step() const
    {
      return m_work_step;
    }

    const std::string& work_step_desc() const
    {
      return m_work_step_desc;
    }
    
    const std::string& work_step_unit() const
    {
      return m_work_step_unit;
    }

  protected:
    void add_progress(size_t n_items)
    {
      std::atomic_fetch_add(&m_work_completed, n_items);
    }
    
    void next_step(size_t n_items, std::string_view desc, std::string_view unit)
    {
      ++m_work_step;
      m_work_step_desc = desc;
      m_work_step_unit = unit;
      m_work_total = n_items;
      m_work_completed.store(0);
    }

  private:
    virtual tf::Future<RetT> run_async(mpcf::Executor& exec) = 0;
    
    tf::Future<RetT> m_future;
    std::atomic_bool m_stop_requested = false;
    
    size_t m_work_total = 0ul;
    std::atomic_uint64_t m_work_completed;

    size_t m_work_step = 0ul;
    std::string m_work_step_desc;
    std::string m_work_step_unit;
  };
}

#endif
