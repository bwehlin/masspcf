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
      std::cout << "Starting async..." << std::endl;
      m_stop_requested.store(false);
      m_future = run_async(exec);
      return *this;
    }
    
  private:
    virtual std::future<RetT> run_async(mpcf::Executor& exec) = 0;
    
    std::future<RetT> m_future;
    std::atomic_bool m_stop_requested = false;
  };
}

#endif
