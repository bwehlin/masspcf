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

#ifndef MPCF_CUDA_STREAM_H
#define MPCF_CUDA_STREAM_H

#include <cuda_runtime.h>

#include <mpcf/cuda/cuda_util.cuh>

namespace mpcf
{
  /// Move-only RAII wrapper around cudaStream_t. Default ctor creates
  /// via cudaStreamCreate; dtor destroys. Mirrors CudaDeviceArray's
  /// ownership model so the two compose naturally.
  class CudaStream
  {
  public:
    CudaStream() { CHK_CUDA(cudaStreamCreate(&m_stream)); }

    ~CudaStream() noexcept
    {
      if (m_stream)
      {
        cudaStreamDestroy(m_stream);
      }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
      : m_stream(other.m_stream)
    {
      other.m_stream = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept
    {
      if (this != &other)
      {
        if (m_stream)
        {
          cudaStreamDestroy(m_stream);
        }
        m_stream = other.m_stream;
        other.m_stream = nullptr;
      }
      return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return m_stream; }
    [[nodiscard]] cudaStream_t* ptr() noexcept { return &m_stream; }

  private:
    cudaStream_t m_stream = nullptr;
  };
}

#endif
