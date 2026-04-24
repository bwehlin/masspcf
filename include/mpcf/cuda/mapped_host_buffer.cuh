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

#ifndef MPCF_CUDA_MAPPED_HOST_BUFFER_H
#define MPCF_CUDA_MAPPED_HOST_BUFFER_H

#include <cuda_runtime.h>

#include <mpcf/cuda/cuda_util.cuh>

#include <cstddef>
#include <iostream>

namespace mpcf
{
  /// Move-only RAII wrapper for cudaHostAlloc'd memory with the
  /// Portable | Mapped flags: the buffer is page-locked on the host
  /// AND mapped into the CUDA device address space, so the same
  /// allocation is reachable from host code (via data()) and from
  /// kernel code (via device_ptr()) without a D2H/H2D copy.
  ///
  /// Used for small counters that kernels increment via atomicAdd and
  /// the host reads immediately after a stream synchronize.
  template <typename T>
  class MappedHostBuffer
  {
  public:
    MappedHostBuffer() = default;

    explicit MappedHostBuffer(std::size_t count)
      : m_count(count)
    {
      if (count > 0)
      {
        CHK_CUDA(cudaHostAlloc(
          reinterpret_cast<void**>(&m_host),
          count * sizeof(T),
          cudaHostAllocPortable | cudaHostAllocMapped));
        CHK_CUDA(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&m_device), m_host, 0));
      }
    }

    ~MappedHostBuffer() noexcept { free_buffer(); }

    MappedHostBuffer(const MappedHostBuffer&) = delete;
    MappedHostBuffer& operator=(const MappedHostBuffer&) = delete;

    MappedHostBuffer(MappedHostBuffer&& other) noexcept
      : m_host(other.m_host), m_device(other.m_device), m_count(other.m_count)
    {
      other.m_host = nullptr;
      other.m_device = nullptr;
      other.m_count = 0;
    }

    MappedHostBuffer& operator=(MappedHostBuffer&& rhs) noexcept
    {
      if (this != &rhs)
      {
        free_buffer();
        m_host = rhs.m_host;
        m_device = rhs.m_device;
        m_count = rhs.m_count;
        rhs.m_host = nullptr;
        rhs.m_device = nullptr;
        rhs.m_count = 0;
      }
      return *this;
    }

    [[nodiscard]] T* data() noexcept { return m_host; }
    [[nodiscard]] const T* data() const noexcept { return m_host; }
    [[nodiscard]] std::size_t size() const noexcept { return m_count; }

    [[nodiscard]] T& operator*() noexcept { return *m_host; }
    [[nodiscard]] const T& operator*() const noexcept { return *m_host; }

    /// Mapped device pointer for the same buffer. Cached at allocation
    /// time so call sites can use it on every kernel launch.
    [[nodiscard]] T* device() const noexcept { return m_device; }

  private:
    void free_buffer() noexcept
    {
      if (!m_host) return;
      cudaError_t rv = cudaFreeHost(m_host);
      if (rv != cudaSuccess)
      {
        std::cerr << "Warning: cudaFreeHost failed for "
                  << m_count * sizeof(T) << " bytes" << std::endl;
      }
      m_host = nullptr;
      m_device = nullptr;
      m_count = 0;
    }

    T* m_host = nullptr;
    T* m_device = nullptr;
    std::size_t m_count = 0;
  };
}

#endif
