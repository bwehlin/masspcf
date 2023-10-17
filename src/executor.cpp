#include <mpcf/executor.h>

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include "cuda_util.h"

mpcf::Executor::Executor(Hardware hw, size_t nThreads)
  : m_ownsTfExec(true)
  , m_tfExec(new tf::Executor(nThreads))
  , m_hw(hw)
{
  
}

mpcf::Executor::~Executor()
{
  if (m_ownsTfExec)
  {
    delete m_tfExec;
  }
}

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

mpcf::Executor& mpcf::default_executor(mpcf::Hardware hw)
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
