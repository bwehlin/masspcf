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
    
    CudaDeviceArray(std::size_t sz)
    {
      allocate(sz);
    }
    
    CudaDeviceArray(const std::vector<T>& data)
    {
      allocate(data.size());
      toDevice(data);
    }
    
    CudaDeviceArray(CudaDeviceArray&& other)
      : m_devPtr(other.m_devPtr)
      , m_sz(other.m_sz)
    {
      other.m_devPtr = nullptr;
      other.m_sz = 0ul;
    }
    
    CudaDeviceArray& operator=(CudaDeviceArray&& rhs)
    {
      if (&rhs == this)
      {
        return *this;
      }
      m_devPtr = rhs.m_devPtr;
      m_sz = rhs.m_sz;
      rhs.m_devPtr = nullptr;
      rhs.m_sz = 0;
      return *this;
    }
    
    ~CudaDeviceArray()
    {
      if (!m_devPtr)
      {
        return;
      }
      auto rv = cudaFree(m_devPtr);
      if (rv != cudaSuccess)
      {
        std::cout << "Warning! Could not deallocate " << m_sz * sizeof(T) << " byte array stored on GPU";
      }
    }
    
    CudaDeviceArray(const CudaDeviceArray&) = delete;
    CudaDeviceArray& operator=(const CudaDeviceArray&) = delete;
    
    T* get() const { return m_devPtr; }
    
    void toDevice(const T* src, std::size_t n)
    {
      assertSz(n);
      CHK_CUDA(cudaMemcpy(m_devPtr, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void toDevice(const std::vector<T>& src)
    {
      toDevice(src.data(), src.size());
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
    
  private:
    void assertSz(std::size_t n) const
    {
      if (n > m_sz)
      {
        throw std::runtime_error("Tried copying " + std::to_string(n) + " element(s) to/from array of size " + std::to_string(m_sz));
      }
    }
    
    void allocate(std::size_t sz)
    {
      CHK_CUDA(cudaMalloc(&m_devPtr, sizeof(T) * sz));
      m_sz = sz;
    }
    
    std::mutex m_mutex;
    T* m_devPtr = nullptr;
    std::size_t m_sz = 0ul;
  };
}

#endif
