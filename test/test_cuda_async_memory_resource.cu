/*
* Copyright 2024-2026 Bjorn Wehlin
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

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <mpcf/cuda/cuda_async_memory_resource.cuh>
#include <mpcf/cuda/cuda_util.cuh>

#include <cstddef>
#include <limits>

// On OOM, CudaAsyncMemoryResource must throw mpcf::cuda_error (not
// thrust::bad_alloc). RipserPlusPlusTask::dispatch_item catches only
// mpcf::cuda_error with code cudaErrorMemoryAllocation to trigger its
// AIMD + CPU fallback; any thrust-shaped exception would unwind past
// the catch and abort the entire batch. This guards that contract at
// the resource boundary so the fallback path actually fires.
TEST(CudaAsyncMemoryResource, OomThrowsMpcfCudaErrorWithAllocationCode)
{
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  mpcf::CudaAsyncMemoryResource mr(stream);

  // Request a size that no current device can satisfy. cudaMallocAsync
  // returns cudaErrorMemoryAllocation in that case.
  constexpr std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;

  bool caught_mpcf = false;
  cudaError_t observed_code = cudaSuccess;
  try {
    auto p = mr.do_allocate(huge);
    // Unexpected success: clean up so we don't leak before failing.
    mr.do_deallocate(p, huge, THRUST_MR_DEFAULT_ALIGNMENT);
    ADD_FAILURE() << "Expected allocation to fail";
  }
  catch (const mpcf::cuda_error& e) {
    caught_mpcf = true;
    observed_code = e.code();
  }
  catch (const std::exception& e) {
    ADD_FAILURE() << "Expected mpcf::cuda_error, got: " << e.what();
  }
  catch (...) {
    ADD_FAILURE() << "Expected mpcf::cuda_error, got unknown exception";
  }

  EXPECT_TRUE(caught_mpcf);
  EXPECT_EQ(observed_code, cudaErrorMemoryAllocation);

  // Clear any sticky runtime error left by the failed allocation.
  cudaGetLastError();
  cudaStreamDestroy(stream);
}
