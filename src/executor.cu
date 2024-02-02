#include <mpcf/executor.h>
#include <taskflow/taskflow.hpp>

#include <stdexcept>

#ifdef BUILD_WITH_CUDA
#pragma message("Building mpcf_cpp with CUDA")
#else
#pragma message("Building mpcf_cpp without CUDA")
#endif

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <mpcf/cuda/cuda_util.cuh>
#endif

size_t mpcf::get_num_cuda_devices()
{
  int nGpus = 0;
#ifdef BUILD_WITH_CUDA
  if (cudaGetDeviceCount(&nGpus) != cudaSuccess)
  {
    return 0;
  }
  if (nGpus < 0)
  {
    // Just in case...
    throw std::runtime_error("Negative number (" + std::to_string(nGpus) + ") of GPUs reported!");
  }
#endif
  return static_cast<size_t>(nGpus);
}

mpcf::Executor& mpcf::default_executor()
{
  static Executor exec = Executor(std::thread::hardware_concurrency(), get_num_cuda_devices());
  return exec;
}
