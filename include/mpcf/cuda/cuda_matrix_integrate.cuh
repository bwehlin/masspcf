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


#ifndef MPCF_CUDA_MATRIX_INTEGRATE_CUH
#define MPCF_CUDA_MATRIX_INTEGRATE_CUH

#include <cuda_runtime.h>

#include "../task.h"
#include "../pcf.h"
#include "../executor.h"
#include "../operations.cuh"
#include "../block_matrix_support.cuh"

#include "cuda_util.cuh"
#include "cuda_matrix_integrate_structs.cuh"
#include "cuda_device_array.cuh"

#include "cuda_runtime.h"

#include <taskflow/taskflow.hpp>

#include <vector>
#include <iostream>

namespace mpcf
{
  namespace internal
  {
    // Return the maximum number of T's that can be allocated on a single GPU
    template <typename T>
    size_t get_max_allocation_n(size_t nGpus) // Use first nGpus GPUs
    {
      constexpr float allocationPct = 0.8f; // Use at most this percentage of free GPU ram for matrix (leave some space for other stuff)

      size_t retVal = std::numeric_limits<size_t>::max();

      for (size_t i = 0; i < nGpus; ++i)
      {
        CHK_CUDA(cudaSetDevice(static_cast<int>(i)));

        size_t freeMem;
        size_t totalMem;
        CHK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));

        size_t maxMatrixAllocSz = static_cast<size_t>(static_cast<float>(freeMem) * allocationPct);
        size_t maxAllocationN = maxMatrixAllocSz / sizeof(float);
        maxAllocationN = (maxAllocationN / 1024) * 1024;

        retVal = std::min(retVal, maxAllocationN);
      }

      return retVal;
    }

    template <typename T>
    std::vector<std::pair<size_t, size_t>> get_block_row_boundaries(size_t nGpus, size_t nPcfs)
    {
      auto maxAllocationN = get_max_allocation_n<T>(nGpus);
      auto nSplits = nGpus * 32; // Give the scheduler something to work with
      auto rowHeight = mpcf::internal::get_row_size(maxAllocationN, nSplits, nPcfs);
      return mpcf::subdivide(rowHeight, nPcfs);
    }

    template <typename Tt, typename Tv>
    struct DeviceKernelParams
    {
      Tv* matrix;
      size_t* timePointOffsets;
      SimplePoint<Tt, Tv>* points;
      size_t nPcfs;
    };

    struct RowInfo
    {
      size_t rowStart;
      size_t rowHeight;
      size_t iRow;
    };

    template<typename Tt, typename Tv, typename RectangleCallback>
    __device__ void cuda_iterate_rectangles(const DeviceKernelParams<Tt, Tv>& params, size_t fMatrixIdx, size_t gMatrixIdx, Tt a, Tt b, RectangleCallback cb)
    {
      Tt t = a;
      Tt tprev = t;

      Tv fv;
      Tv gv;

      size_t fi = 0; // max_time_prior_to(f, a);
      size_t gi = 0; // max_time_prior_to(g, a);

      size_t fOffset = params.timePointOffsets[fMatrixIdx];
      size_t gOffset = params.timePointOffsets[gMatrixIdx];

      size_t fsz = params.timePointOffsets[fMatrixIdx + 1] - fOffset;
      size_t gsz = params.timePointOffsets[gMatrixIdx + 1] - gOffset;

      mpcf::internal::SimplePoint<Tt, Tv>* fpts = params.points + fOffset;
      mpcf::internal::SimplePoint<Tt, Tv>* gpts = params.points + gOffset;

      while (t < b)
      {
        tprev = t;
        fv = fpts[fi].v;
        gv = gpts[gi].v;

        if (fi + 1 < fsz && gi + 1 < gsz)
        {
          auto delta = fpts[fi + 1].t - gpts[gi + 1].t;
          if (delta <= 0)
          {
            ++fi;
          }
          if (delta >= 0)
          {
            ++gi;
          }
        }
        else
        {
          if (fi + 1 < fsz)
          {
            ++fi;
          }
          else if (gi + 1 < gsz)
          {
            ++gi;
          }
          else
          {
            cb(tprev, b, fv, gv);
            return;
          }
        }

        t = max(fpts[fi].t, gpts[gi].t);
        cb(tprev, t, fv, gv);
      }
    }

    template <typename Tt, typename Tv, typename ComboOp>
    __global__
      void cuda_riemann_integrate(
        DeviceKernelParams<Tt, Tv> params,
        RowInfo rowInfo, Tt a, Tt b, ComboOp op)
    {
      size_t iBlock = blockDim.x * blockIdx.x + threadIdx.x;
      size_t j = blockDim.y * blockIdx.y + threadIdx.y;

      if (iBlock >= rowInfo.rowHeight)
      {
        return;
      }

      size_t i = iBlock + rowInfo.rowStart;

      if (j < i || i >= params.nPcfs || j >= params.nPcfs)
      {
        return;
      }

      Tv ret = 0;
      cuda_iterate_rectangles<Tt, Tv>(params, i, j, a, b, [&ret, op](Tt l, Tt r, Tv f, Tv g) {
        ret += (r - l) * op(f, g);
        });

      params.matrix[iBlock * params.nPcfs + j] = op(ret);
    }

    // For some reason, calling cuda_riemann_integrate directly from within a class raises a syntax error on MSVC
    template <typename Tt, typename Tv, typename ComboOp>
    void call_riemann_integrate(dim3 gridDim, dim3 blockDim, const DeviceKernelParams<Tt, Tv>& params, const RowInfo& rowInfo, Tt a, Tt b, ComboOp op)
    {
      cuda_riemann_integrate<Tt, Tv, ComboOp> << <gridDim, blockDim >> > (params, rowInfo, a, b, op);
    }

    template <typename PcfFwdIt, typename ComboOp, typename ProgressCb = std::function<void(size_t)>>
    class CudaMatrixRectangleIterator
    {
    public:
      using pcf_type = typename PcfFwdIt::value_type;
      using time_type = typename pcf_type::time_type;
      using value_type = typename pcf_type::value_type;

      CudaMatrixRectangleIterator(tf::Executor& exec, value_type* out, PcfFwdIt begin, PcfFwdIt end, ProgressCb progressCb = [](size_t) {})
        : m_out(out)
        , m_gpuHostThreads(exec)
        , m_progressCb(progressCb)
      {
        m_nGpus = exec.num_workers();
        m_canceled.store(false);
        //init(begin, end);
      }

      void riemann_integrate(time_type a, time_type b, ComboOp op)
      {
        for_each_block_row([this, a, b, op](dim3 gridDim, dim3 blockDim, const DeviceKernelParams<time_type, value_type>& params, const RowInfo& rowInfo) {
          call_riemann_integrate<time_type, value_type>(gridDim, blockDim, params, rowInfo, a, b, op);
          });
      }
      
      void cancel()
      {
        m_canceled.store(true);
      }

      void set_block_dim(const dim3& dim)
      {
        m_blockDim = dim;
      }
      
      void init(PcfFwdIt begin, PcfFwdIt end)
      {
        m_nPcfs = std::distance(begin, end);

        init_host_offset_data(begin, end);
        init_block_row_boundaries();
        init_device_storages();
        copy_offset_data_to_devices();
      }
      
    private:
      void init_host_offset_data(PcfFwdIt begin, PcfFwdIt end)
      {
        m_h_offsetData.timePointOffsets.resize(m_nPcfs + 1);

        size_t i = 0ul;
        // Compute size required for all PCFs
        size_t offset = 0ul;
        for (auto it = begin; it != end; ++it)
        {
          auto const& f = (*it).points();
          m_h_offsetData.timePointOffsets[i++] = offset;
          offset += f.size();
        }

        // Store PCFs
        m_h_offsetData.points.resize(offset);
        i = 0ul;
        for (auto it = begin; it != end; ++it)
        {
          auto const& f = (*it).points();
          auto csz = f.size();
          auto coffs = m_h_offsetData.timePointOffsets[i++];
          for (auto j = 0ul; j < csz; ++j)
          {
            m_h_offsetData.points[coffs + j].t = f[j].t;
            m_h_offsetData.points[coffs + j].v = f[j].v;
          }
        }

        m_h_offsetData.timePointOffsets[m_nPcfs] = m_h_offsetData.timePointOffsets[m_nPcfs - 1] + (*std::prev(end)).points().size();
      }

      void init_block_row_boundaries()
      {
        m_blockRowBoundaries = mpcf::internal::get_block_row_boundaries<value_type>(m_nGpus, m_nPcfs);
      }

      void init_device_storages()
      {
        m_deviceStorages.resize(m_nGpus);
        auto maxRowHeight = m_blockRowBoundaries[0].second + 1;

        for (size_t iGpu = 0; iGpu < m_nGpus; ++iGpu)
        {
          init_device_storage(iGpu, maxRowHeight);
        }
      }

      void init_device_storage(size_t iGpu, size_t rowHeight)
      {
        auto& storage = m_deviceStorages[iGpu];

        CHK_CUDA(cudaSetDevice(static_cast<int>(iGpu)));

        storage.matrix = mpcf::CudaDeviceArray<value_type>(rowHeight * m_nPcfs);
        storage.points = mpcf::CudaDeviceArray<mpcf::internal::SimplePoint<time_type, value_type>>(m_h_offsetData.points);
        storage.timePointOffsets = mpcf::CudaDeviceArray<size_t>(m_h_offsetData.timePointOffsets);
      }

      void copy_offset_data_to_devices()
      {
        for (size_t iGpu = 0; iGpu < m_nGpus; ++iGpu)
        {
          CHK_CUDA(cudaSetDevice(static_cast<int>(iGpu)));
          m_deviceStorages[iGpu].points.toDevice(m_h_offsetData.points);
          m_deviceStorages[iGpu].timePointOffsets.toDevice(m_h_offsetData.timePointOffsets);
        }
      }

      size_t get_row_height_from_boundaries(std::pair<size_t, size_t> boundaries)
      {
        return boundaries.second - boundaries.first + 1;
      }

      DeviceKernelParams<time_type, value_type> make_kernel_params(size_t iGpu) const
      {
        DeviceKernelParams<time_type, value_type> params;
        auto& storage = m_deviceStorages[iGpu];

        params.matrix = storage.matrix.get();
        params.points = storage.points.get();
        params.timePointOffsets = storage.timePointOffsets.get();

        params.nPcfs = m_nPcfs;

        return params;
      }

      /// Run the supplied kernel launch function on each block row. 'for_each_block_row' blocks until the whole operation is complete.
      void for_each_block_row(std::function<void(dim3, dim3, const DeviceKernelParams<time_type, value_type>&, const RowInfo&)> launchFunc)
      {
        tf::Taskflow flow;

        flow.for_each_index<size_t, size_t, size_t>(0ul, m_blockRowBoundaries.size(), 1ul, [this, launchFunc](size_t i) {
          if (m_canceled.load())
          {
            return;
          }
          exec_block_row(i, launchFunc);
          });

        m_gpuHostThreads.run(std::move(flow));

        // Keep in mind 'this' runs on a separate thread from main already, so it is OK to block here.
        m_gpuHostThreads.wait_for_all();
      }

      void exec_block_row(size_t iRow, std::function<void(dim3, dim3, const DeviceKernelParams<time_type, value_type>&, const RowInfo&)> launchFunc)
      {
        // This function executes on a CPU thread that drives one GPU

        auto iGpu = m_gpuHostThreads.this_worker_id(); // Worker IDs are guaranteed to be 0...(n-1) for n threads.

        CHK_CUDA(cudaSetDevice(static_cast<int>(iGpu)));

        m_deviceStorages[iGpu].matrix.clear();

        value_type* hostMatrix = m_out;

        auto const& rowBoundaries = m_blockRowBoundaries[iRow];

        size_t rowHeight = get_row_height_from_boundaries(rowBoundaries);
        dim3 gridDim = mpcf::internal::get_grid_dims(m_blockDim, rowHeight, m_nPcfs);

        auto const & params = make_kernel_params(iGpu);
        
        RowInfo rowInfo;
        rowInfo.rowHeight = rowHeight;
        rowInfo.rowStart = rowBoundaries.first;
        rowInfo.iRow = iRow;

        launchFunc(gridDim, m_blockDim, params, rowInfo);
        CHK_CUDA(cudaPeekAtLastError());

        value_type* target = &hostMatrix[rowInfo.rowStart * m_nPcfs];
        auto nEntries = rowHeight * m_nPcfs;

        // These are non-overlapping writes so no need to lock the target
        m_deviceStorages[iGpu].matrix.toHost(target, nEntries);

        auto progress = (rowBoundaries.second - rowBoundaries.first + 1) * (2 * m_nPcfs - rowBoundaries.first - rowBoundaries.second) / 2;
        m_progressCb(progress);
      }
      
      size_t m_nPcfs;

      value_type* m_out;

      tf::Executor& m_gpuHostThreads;
      size_t m_nGpus;

      mpcf::internal::HostPcfOffsetData<time_type, value_type> m_h_offsetData;
      std::vector<std::pair<size_t, size_t>> m_blockRowBoundaries;

      std::vector<mpcf::internal::DeviceStorage<time_type, value_type>> m_deviceStorages;

      dim3 m_blockDim = dim3(32, 1, 1);

      ProgressCb m_progressCb;
      
      std::atomic_bool m_canceled;
    };

  } // namespace internal

  template <typename PcfIt>
  struct PcfIteratorTraits
  {
    using pcf_type = typename PcfIt::value_type;
    using point_type = typename pcf_type::point_type;
    using time_type = typename point_type::time_type;
    using value_type = typename point_type::value_type;
  };

  template <typename PcfFwdIt, typename ComboOp>
  class MatrixIntegrateCudaTask : public mpcf::StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfIteratorTraits<PcfFwdIt>::pcf_type;
    using time_type = typename PcfIteratorTraits<PcfFwdIt>::time_type;
    using value_type = typename PcfIteratorTraits<PcfFwdIt>::value_type;

    MatrixIntegrateCudaTask(tf::Executor& cudaThreads, value_type* out, PcfFwdIt beginPcfs, PcfFwdIt endPcfs, ComboOp op = {}, time_type a = 0, time_type b = std::numeric_limits<time_type>::max())
      : m_fs(beginPcfs, endPcfs)
      , m_op(op)
      , m_a(a)
      , m_b(b)
      , m_out(out)
      , m_iterator(cudaThreads, out, m_fs.begin(), m_fs.end(), [this](size_t n) { add_progress(n); })
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      m_iterator.set_block_dim(get_block_dim());
      m_iterator.init(m_fs.begin(), m_fs.end());
      
      next_step(m_fs.size() * (m_fs.size() + 1) / 2, "Computing upper triangle", "integral");

      tf::Taskflow flow;
      std::vector<tf::Task> tasks;
      
      auto sz = m_fs.size();
      
      tasks.emplace_back(flow.emplace([this] { m_iterator.riemann_integrate(m_a, m_b, m_op); }));
      tasks.emplace_back(flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [this, sz](size_t i) {
        for (size_t j = 0; j < i; ++j)
        {
          m_out[i * sz + j] = m_out[j * sz + i];
        }
      }));
      
      tasks.emplace_back(create_terminal_task(flow));
      flow.linearize(tasks);

      // We run the task as a CPU task. The actual job will spawn additional tasks on the GPU
      // executor.
      return exec.cpu()->run(std::move(flow));
    }
    
    void on_stop_requested() override
    {
      m_iterator.cancel();
    }

    std::vector<pcf_type> m_fs;

    ComboOp m_op;
    time_type m_a;
    time_type m_b;
    value_type* m_out;
    
    internal::CudaMatrixRectangleIterator<typename std::vector<pcf_type>::const_iterator, ComboOp> m_iterator;
  };

  template <typename PcfFwdIt, typename TOperation>
  std::unique_ptr<StoppableTask<void>> create_matrix_integrate_cuda_task(
      typename PcfIteratorTraits<PcfFwdIt>::value_type* out,
      PcfFwdIt beginPcfs,
      PcfFwdIt endPcfs,
      TOperation op,
      typename PcfIteratorTraits<PcfFwdIt>::time_type a = 0,
      typename PcfIteratorTraits<PcfFwdIt>::time_type b = std::numeric_limits<typename PcfIteratorTraits<PcfFwdIt>::time_type>::max())
  {
    return std::make_unique<MatrixIntegrateCudaTask<PcfFwdIt, TOperation>>(*default_executor().cuda(), out, beginPcfs, endPcfs, op, a, b);
  }

}

#endif
