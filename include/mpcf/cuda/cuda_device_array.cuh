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

#ifndef MPCF_CUDA_DEVICE_ARRAY_H
#define MPCF_CUDA_DEVICE_ARRAY_H

#include <cstdint>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>

#include "cuda_util.cuh"

namespace mpcf
{
  template <typename T>
  class CudaDeviceArray
  {
  public:
    CudaDeviceArray() = default;
    
    explicit CudaDeviceArray(std::size_t sz)
    {
      allocate(sz);
    }

    explicit CudaDeviceArray(const std::vector<T>& data)
    {
      allocate(data.size());
      toDevice(data);
    }
    
    CudaDeviceArray(CudaDeviceArray&& other)
      : m_devPtr(other.m_devPtr)
      , m_sz(other.m_sz)
      , m_stream(other.m_stream)
    {
      other.m_devPtr = nullptr;
      other.m_sz = 0ul;
      other.m_stream = nullptr;
    }

    CudaDeviceArray& operator=(CudaDeviceArray&& rhs)
    {
      if (&rhs == this)
      {
        return *this;
      }
      free_buffer();
      m_devPtr = rhs.m_devPtr;
      m_sz = rhs.m_sz;
      m_stream = rhs.m_stream;
      rhs.m_devPtr = nullptr;
      rhs.m_sz = 0;
      rhs.m_stream = nullptr;
      return *this;
    }

    ~CudaDeviceArray()
    {
      free_buffer();
    }
    
    CudaDeviceArray(const CudaDeviceArray&) = delete;
    CudaDeviceArray& operator=(const CudaDeviceArray&) = delete;

    /// (Re)allocate the array to the given size. Frees any existing
    /// allocation first, so calling repeatedly does not leak.
    void allocate(std::size_t sz)
    {
      reset();
      CHK_CUDA(cudaMalloc(&m_devPtr, sizeof(T) * sz));
      m_sz = sz;
      m_stream = nullptr;
    }

    /// Stream-ordered (re)allocate via the current device's default
    /// memory pool. Uses cudaMallocAsync, so the allocation is visible
    /// to work already queued on `stream` and subsequent work on the
    /// same stream; other streams must synchronize via an event before
    /// using the pointer. Pool-backed so repeated alloc/free cycles
    /// are cheap (the pool caches memory up to its release threshold;
    /// see GpuMemoryScheduler's ctor which sets that threshold high).
    /// The stream is stored so the destructor / reset can issue a
    /// matching cudaFreeAsync; the caller must guarantee the stream
    /// outlives this array (or call reset() explicitly before the
    /// stream is destroyed).
    void allocate_async(std::size_t sz, cudaStream_t stream)
    {
      reset();
      CHK_CUDA(cudaMallocAsync(&m_devPtr, sizeof(T) * sz, stream));
      m_sz = sz;
      m_stream = stream;
    }

    /// Free the underlying buffer and reset to empty. No-op if already empty.
    void reset() noexcept
    {
      free_buffer();
    }

    T* get() const { return m_devPtr; }
    std::size_t size() const noexcept { return m_sz; }

    /// Implicit conversion to the raw device pointer. Lets instances be
    /// used in kernel launches, thrust iterators, and cudaMemcpy calls
    /// without an explicit .get() at every site.
    operator T*() const noexcept { return m_devPtr; }
    T* operator->() const noexcept { return m_devPtr; }

    /// Address of the internal device pointer. Stable for the lifetime of
    /// the CudaDeviceArray instance. Used when copying the pointer value
    /// itself into device memory (e.g., to wire up a nested pointer field
    /// inside a device struct via cudaMemcpy).
    T* const* address_of_ptr() const noexcept { return &m_devPtr; }

    void toDevice(const T* src, std::size_t n)
    {
      assertSz(n);
      CHK_CUDA(cudaMemcpy(m_devPtr, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void toDevice(const std::vector<T>& src)
    {
      toDevice(src.data(), src.size());
    }

    void toDeviceAsync(cudaStream_t stream, const T* src, std::size_t n)
    {
      assertSz(n);
      CHK_CUDA(cudaMemcpyAsync(m_devPtr, src, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    
    void toHost(T* dst, size_t nElems = 0ul)
    {
      if (nElems == 0ul)
      {
        nElems = m_sz;
      }
      CHK_CUDA(cudaMemcpy(dst, m_devPtr, nElems * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void toHostAsync(cudaStream_t stream, T* dst, size_t nElems = 0ul, size_t offset = 0ul)
    {
      if (nElems == 0ul)
      {
        nElems = m_sz;
      }
      CHK_CUDA(cudaMemcpyAsync(dst, m_devPtr + offset, nElems * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    
    void toHost(std::vector<T>& dst)
    {
      dst.resize(m_sz);
      toHost(dst.data());
    }
    
    void clear()
    {
      CHK_CUDA(cudaMemset(m_devPtr, 0, m_sz * sizeof(T)));
    }

    void clearAsync(cudaStream_t stream)
    {
      CHK_CUDA(cudaMemsetAsync(m_devPtr, 0, m_sz * sizeof(T), stream));
    }
    
  private:
    void assertSz(std::size_t n) const
    {
      if (n > m_sz)
      {
        throw std::runtime_error("Tried copying " + std::to_string(n) + " element(s) to/from array of size " + std::to_string(m_sz));
      }
    }

    /// Release the current buffer using the matching free path:
    /// cudaFreeAsync when allocate_async was used (stream tracked),
    /// cudaFree when allocate was used. Safe to call on an already-
    /// empty array. Resets internal state to empty.
    void free_buffer() noexcept
    {
      if (!m_devPtr)
      {
        return;
      }
      cudaError_t rv = cudaSuccess;
      if (m_stream)
      {
        rv = cudaFreeAsync(m_devPtr, m_stream);
      }
      else
      {
        rv = cudaFree(m_devPtr);
      }
      if (rv != cudaSuccess)
      {
        // Drop the runtime's sticky error so unrelated callers of
        // cudaGetLastError don't see it. If freeing via the async
        // path failed (e.g., stream already destroyed), fall back
        // to sync cudaFree so we don't leak.
        cudaGetLastError();
        if (m_stream)
        {
          cudaFree(m_devPtr);
        }
      }
      m_devPtr = nullptr;
      m_sz = 0;
      m_stream = nullptr;
    }

    std::mutex m_mutex;
    T* m_devPtr = nullptr;
    std::size_t m_sz = 0ul;
    cudaStream_t m_stream = nullptr;
  };
}

#endif
