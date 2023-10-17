#ifndef MPCF_EXECUTOR_H
#define MPCF_EXECUTOR_H

//#include <taskflow/core/executor.hpp>
#include <taskflow/taskflow.hpp>
#include <memory>

namespace mpcf
{
  enum class Hardware
  {
    CPU,
    CUDA
  };
  
  class Executor
  {
  public:
    ~Executor() noexcept;
    
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    
    Executor(Executor&& other) noexcept
      : m_ownsTfExec(other.m_ownsTfExec)
      , m_tfExec(other.m_tfExec)
      , m_hw(other.m_hw)
    {
      other.m_ownsTfExec = false;
      other.m_tfExec = nullptr;
    }
    
    Executor& operator=(Executor&& rhs) noexcept
    {
      if (this == &rhs)
      {
        return *this;
      }
      
      m_ownsTfExec = rhs.m_ownsTfExec;
      m_tfExec = rhs.m_tfExec;
      m_hw = rhs.m_hw;
      
      rhs.m_ownsTfExec = false;
      rhs.m_tfExec = nullptr;
      
      return *this;
    }
    
    static Executor create_cpu(size_t nThreads = std::thread::hardware_concurrency())
    {
      return Executor(Hardware::CPU, nThreads);
    }
    
    /// Create a CPU executor that is using an existing TaskFlow executor. It is the responsibility
    /// of the client to keep the TaskFlow executor alive for the duration of the Executor
    static Executor create_cpu(tf::Executor& tfExec)
    {
      return Executor(&tfExec);
    }
    
#ifdef BUILD_WITH_CUDA
    static Executor create_cuda();
#endif
    
    Hardware hardware() const { return m_hw; }
    
    tf::Executor* operator->() noexcept
    {
      return m_tfExec;
    }
    
  private:
    Executor(tf::Executor* tfExec) noexcept
      : m_ownsTfExec(false)
      , m_tfExec(tfExec)
      , m_hw(Hardware::CPU)
    {
      
    }
    
    Executor(Hardware hw, size_t nThreads);
    
    bool m_ownsTfExec;
    tf::Executor* m_tfExec;
    Hardware m_hw;
  };
  
  Executor& default_cpu_executor();
  
#ifdef BUILD_WITH_CUDA
  Executor& default_cuda_executor();
#endif
  
  Executor& default_executor(Hardware hw);
}

#endif
