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

// This file is compiled by NVCC.  It pulls in the full CUDA kernel code
// and provides concrete (non-template) factory functions that the rest
// of the project can call without requiring NVCC.

#include <mpcf/cuda/cuda_matrix_integrate_api.h>
#include <mpcf/cuda/cuda_matrix_integrate.cuh>
#include <mpcf/operations.cuh>

namespace mpcf
{
  std::unique_ptr<StoppableTask<void>> create_cuda_matrix_integrate_l1_task(
      float32_t* out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a, float32_t b)
  {
    using iter_t = std::vector<Pcf<float32_t, float32_t>>::const_iterator;
    using op_t = OperationL1Dist<float32_t, float32_t>;

    return std::make_unique<MatrixIntegrateCudaTask<iter_t, op_t>>(
        *default_executor().cuda(), out, pcfs.cbegin(), pcfs.cend(), op_t{}, a, b);
  }

  std::unique_ptr<StoppableTask<void>> create_cuda_matrix_integrate_l1_task(
      float64_t* out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a, float64_t b)
  {
    using iter_t = std::vector<Pcf<float64_t, float64_t>>::const_iterator;
    using op_t = OperationL1Dist<float64_t, float64_t>;

    return std::make_unique<MatrixIntegrateCudaTask<iter_t, op_t>>(
        *default_executor().cuda(), out, pcfs.cbegin(), pcfs.cend(), op_t{}, a, b);
  }
}
