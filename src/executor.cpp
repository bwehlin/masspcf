#include <mpcf/executor.h>

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include "cuda_util.h"

#include <iostream>

mpcf::Executor::Executor(int hw, unsigned long nThreads, bool)
  : m_ownsTfExec(true)
  , m_tfExec(new tf::Executor(nThreads))
  , m_hw(hw)
{ }

// mpcf::Executor::~Executor() noexcept
// {
//   if (m_ownsTfExec)
//   {
//     delete m_tfExec;
//   }
// }

mpcf::Executor
mpcf::Executor::create_cuda()
{
  int nGpus;
  CHK_CUDA(cudaGetDeviceCount(&nGpus));
  
  return Executor(Hardware::CUDA, static_cast<size_t>(nGpus));
}

mpcf::Executor& mpcf::default_cuda_executor()
{
  static Executor exec = Executor::create_cuda();
  return exec;
}

#endif

mpcf::Executor& mpcf::default_cpu_executor()
{
  static Executor exec = Executor::create_cpu();
  return exec;
}

mpcf::Executor& mpcf::default_executor(int hw)
{
  switch (hw)
  {
#ifdef BUILD_WITH_CUDA
  case Hardware::CUDA:
    return default_cuda_executor();
#endif
  default:
    return default_cpu_executor();
  }
}

mpcf::Executor::Executor(Executor&& other) noexcept
  : m_ownsTfExec(other.m_ownsTfExec)
  , m_tfExec(other.m_tfExec)
  , m_hw(other.m_hw)
{
  other.m_ownsTfExec = false;
  other.m_tfExec = nullptr;
}

mpcf::Executor& mpcf::Executor::operator=(Executor&& rhs) noexcept
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

mpcf::Executor mpcf::Executor::create_cpu(size_t nThreads)
{
  return Executor(Hardware::CPU, nThreads);
}

/// Create a CPU executor that is using an existing TaskFlow executor. It is the responsibility
/// of the client to keep the TaskFlow executor alive for the duration of the Executor
mpcf::Executor mpcf::Executor::create_cpu(tf::Executor& tfExec)
{
  return Executor(&tfExec);
}

mpcf::Hardware mpcf::Executor::hardware() const { return m_hw; }

tf::Executor* mpcf::Executor::operator->() noexcept
{
  return m_tfExec;
}

namespace mpcf {
Executor::Executor(tf::Executor* tfExec) noexcept
  : m_ownsTfExec(false)
  , m_tfExec(tfExec)
  , m_hw(Hardware::CPU)
{
  
}
}

mpcf::Executor exec1(mpcf::Hardware::CPU, 1);
auto exec = mpcf::Executor::create_cpu(1);