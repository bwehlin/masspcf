/*
* Copyright 2024 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

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
