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

// Pure C++ header — no NVCC required.
// Declares factory functions for CUDA-accelerated matrix integration tasks.
// Implementations live in src/cuda/cuda_matrix_integrate.cu (compiled by NVCC).

#ifndef MPCF_CUDA_MATRIX_INTEGRATE_API_H
#define MPCF_CUDA_MATRIX_INTEGRATE_API_H

#include "../task.hpp"
#include "../functional/pcf.hpp"

#include <vector>
#include <memory>

namespace mpcf
{
  // L1 distance — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_matrix_integrate_l1_task(
      float32_t* out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // L1 distance — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_matrix_integrate_l1_task(
      float64_t* out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());

  // L2 inner product — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_matrix_integrate_l2_kernel_task(
      float32_t* out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // L2 inner product — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_matrix_integrate_l2_kernel_task(
      float64_t* out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());
}

#endif
