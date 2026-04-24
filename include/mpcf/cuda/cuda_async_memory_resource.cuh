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

#ifndef MPCF_CUDA_ASYNC_MEMORY_RESOURCE_H
#define MPCF_CUDA_ASYNC_MEMORY_RESOURCE_H

#include <cuda_runtime.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/cuda/error.h>

#include <mpcf/cuda/cuda_util.cuh>

#include <cstddef>
#include <new>

namespace mpcf
{
  /// Thrust memory resource backed by cudaMallocAsync / cudaFreeAsync
  /// on a caller-provided CUDA stream.
  ///
  /// Why: thrust's stock `thrust::system::cuda::memory_resource` uses
  /// the synchronous `cudaMalloc`/`cudaFree` for all temp-storage
  /// allocations done by `thrust::sort`, `thrust::count`, etc. With
  /// many concurrent thrust calls (e.g. one per Ripser++ stream)
  /// those serialise on the driver allocator lock and dominate wall
  /// time. Routing them through `cudaMallocAsync` lets the device's
  /// default memory pool service them as a stream-ordered pool-local
  /// bookkeeping op once the pool is warm.
  ///
  /// Pair with `GpuMemoryScheduler`'s ctor, which raises the default
  /// pool's release threshold so freed blocks stay cached up to the
  /// high-water mark instead of returning to the OS.
  ///
  /// Lifetime: holds a non-owning `cudaStream_t`. The caller must
  /// guarantee the stream outlives any thrust call that took an
  /// execution policy referencing this resource. `RipserPlusPlusTask`
  /// satisfies this naturally: each ripser instance owns both its
  /// stream and this resource as members and finishes all thrust
  /// calls before its destructor runs.
  ///
  /// Usage:
  /// ```
  /// CudaAsyncMemoryResource mr(stream);
  /// thrust::sort(thrust::cuda::par(&mr).on(stream), first, last, cmp);
  /// ```
  class CudaAsyncMemoryResource final
    : public thrust::mr::memory_resource<thrust::cuda::pointer<void>>
  {
  public:
    using pointer = thrust::cuda::pointer<void>;

    /// Default-constructed resource holds the null stream; rebind via
    /// `set_stream` once the owning stream is available.
    CudaAsyncMemoryResource() = default;

    explicit CudaAsyncMemoryResource(cudaStream_t stream) noexcept
      : m_stream(stream) {}

    void set_stream(cudaStream_t stream) noexcept { m_stream = stream; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_stream; }

    pointer do_allocate(std::size_t bytes,
                        [[maybe_unused]] std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
      void* ret = nullptr;
      cudaError_t status = cudaMallocAsync(&ret, bytes, m_stream);
      if (status != cudaSuccess)
      {
        // Drop the runtime's sticky error so unrelated callers of
        // cudaGetLastError don't see it.
        cudaGetLastError();
        // Route OOM through mpcf::cuda_error so the hybrid dispatcher's
        // AIMD + CPU fallback catch in compute_persistence.hpp matches.
        // A thrust-shaped bad_alloc would bypass the catch and unwind
        // the whole batch.
        if (status == cudaErrorMemoryAllocation)
        {
          throw ::mpcf::cuda_error(__FILE__, __LINE__, status);
        }
        // Non-OOM failures keep thrust's stock bad_alloc shape; they
        // indicate a programming error (bad stream, bad size) and
        // should not be retried.
        throw thrust::system::detail::bad_alloc(
          thrust::cuda_category().message(status).c_str());
      }
      return pointer(ret);
    }

    void do_deallocate(pointer p,
                       [[maybe_unused]] std::size_t bytes,
                       [[maybe_unused]] std::size_t alignment) override
    {
      void* raw = thrust::detail::pointer_traits<pointer>::get(p);
      // Best-effort free: a failure here would only happen if the
      // stream was destroyed mid-flight, which the lifetime contract
      // forbids; clear sticky error and move on either way.
      if (cudaFreeAsync(raw, m_stream) != cudaSuccess)
      {
        cudaGetLastError();
      }
    }

  private:
    cudaStream_t m_stream = nullptr;
  };
}

#endif
