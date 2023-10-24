#ifndef MPCF_EXECUTOR_H
#define MPCF_EXECUTOR_H

#include <taskflow/core/executor.hpp>
#include <memory>

namespace mpcf
{
  enum class Hardware
  {
    CPU,
    CUDA
  };
  
  size_t get_num_cuda_devices();
  
  /// An Executor instance holds a tf::Executor for CPU-based execution,
  /// and (optionally) a separate tf::Executor for GPU-based execution.
  /// The GPU executor has the same number of threads as there are GPUs,
  /// and each thread operates on only one of the GPUs.
  class Executor
  {
  public:   
    /// Construct an Executor from an existing tf::Executor. It is the
    /// client's responsibility to ensure that the tf::Executor stays
    /// alive for the entire lifetime of this Executor object.
    Executor(tf::Executor* tfExec, size_t nCudaDevices = get_num_cuda_devices()) noexcept
      : m_upCpuExec(nullptr)
      , m_cpuExec(tfExec)
      , m_upCudaExec(nCudaDevices > 0 ? std::make_unique<tf::Executor>(nCudaDevices) : nullptr)
    {
      
    }
    
    Executor(size_t nThreads = std::thread::hardware_concurrency(), size_t nCudaDevices = get_num_cuda_devices())
      : m_upCpuExec(std::make_unique<tf::Executor>(nThreads))
      , m_cpuExec(m_upCpuExec.get())
      , m_upCudaExec(nCudaDevices > 0 ? std::make_unique<tf::Executor>(nCudaDevices) : nullptr)
    {
      
    }
    
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    
    Executor(Executor&& other) noexcept
      : m_upCpuExec(std::move(other.m_cpuExec))
      , m_cpuExec(other.m_cpuExec)
      , m_upCudaExec(std::move(other.m_upCudaExec))
    { }
    
    Executor& operator=(Executor&& rhs) noexcept
    {
      if (this == &rhs)
      {
        return *this;
      }
      
      m_upCpuExec = std::move(rhs.m_upCpuExec);
      m_cpuExec = rhs.m_cpuExec;
      
      m_upCudaExec = std::move(rhs.m_upCudaExec);
      
      return *this;
    }
    
    tf::Executor* cpu() noexcept
    {
      return m_cpuExec;
    }
    
    tf::Executor* cuda() noexcept
    {
      return m_upCudaExec.get();
    }
    
  private:
    std::unique_ptr<tf::Executor> m_upCpuExec;
    tf::Executor* m_cpuExec;
    
    std::unique_ptr<tf::Executor> m_upCudaExec;
  };
  
  Executor& default_executor();
}

#endif
