#ifndef MPCF_TASK_H
#define MPCF_TASK_H

#include <atomic>
#include <future>
#include <functional>
#include <iostream>

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
    }
    
    bool stop_requested() const
    {
      return m_stop_requested.load();
    }
    
    const std::future<RetT>& future() const
    {
      return m_future;
    }
    
    std::future<RetT>& future()
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
    
  protected:
    void add_progress(size_t n_items)
    {
      std::atomic_fetch_add(&m_work_completed, n_items);
    }
    
    void set_total_work(size_t n_items)
    {
      m_work_total = n_items;
    }
    
  private:
    virtual std::future<RetT> run_async(mpcf::Executor& exec) = 0;
    
    std::future<RetT> m_future;
    std::atomic_bool m_stop_requested = false;
    
    size_t m_work_total = 0ul;
    std::atomic_uint64_t m_work_completed;
  };
}

#endif
