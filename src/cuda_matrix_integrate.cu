#include "algorithms/cuda_matrix_integrate.h"

#include "point.h"
#include "cuda_util.h"
#include "cuda_device_array.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <utility>
#include <vector>

#include <taskflow/taskflow.hpp>

namespace
{
  bool verbose = true;
  
  // POD version of Point
  template <typename Tt, typename Tv>
  struct SimplePoint
  {
    Tt t;
    Tv v;
  };
  
  template <typename Tt, typename Tv>
  struct HostPcfOffsetData
  {
    std::vector<size_t> timePointOffsets;
    std::vector<SimplePoint<Tt, Tv>> points;
  };
  
  template <typename Tt, typename Tv>
  struct DeviceStorage
  {
    mpcf::CudaDeviceArray<Tv> matrix;
    mpcf::CudaDeviceArray<size_t> timePointOffsets;
    mpcf::CudaDeviceArray<SimplePoint<Tt, Tv>> points;
    
    DeviceStorage() = default;
    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;
    ~DeviceStorage() = default;
    
    DeviceStorage(DeviceStorage&& other)
      : matrix(std::move(other.matrix))
      , timePointOffsets(std::move(other.timePointOffsets))
      , points(std::move(other.points))
    { }
    
    DeviceStorage& operator=(DeviceStorage&& rhs)
    {
      if (&rhs == this)
      {
        return *this;
      }
      
      matrix = std::move(rhs.matrix);
      timePointOffsets = std::move(rhs.timePointOffsets);
      points = std::move(rhs.points);
      
      return *this;
    }
  };
  
  template <typename Tt, typename Tv>
  struct IntegrationContext
  {
    Tv* hostMatrix;
    
    std::vector<DeviceStorage<Tt, Tv>> deviceStorages;
    HostPcfOffsetData<Tt, Tv> hostOffsetData;
    std::vector<std::pair<size_t, size_t>> blockRowBoundaries;
    
    mpcf::DeviceOp<Tt, Tv> op;
    
    size_t nPcfs;
    
    int nGpus;
    dim3 blockDim;
  };
  
  template <typename Tt, typename Tv>
  HostPcfOffsetData<Tt, Tv>
  get_host_offset_data(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
  {
    HostPcfOffsetData<Tt, Tv> offsetData;
    auto sz = fs.size();
    
    offsetData.timePointOffsets.resize(sz + 1);
    
    // Compute size required for all PCFs
    auto offset = 0ul;
    for (auto i = 0ul; i < sz; ++i)
    {
      auto const & f = fs[i].points();
      offsetData.timePointOffsets[i] = offset;
      offset += f.size();
    }
    
    // Store PCFs
    offsetData.points.resize(offset);
    for (auto i = 0ul; i < sz; ++i)
    {
      auto const & f = fs[i].points();
      auto csz = f.size();
      auto coffs = offsetData.timePointOffsets[i];
      for (auto j = 0ul; j < csz; ++j)
      {
        offsetData.points[coffs + j].t = f[j].t;
        offsetData.points[coffs + j].v = f[j].v;
      }
    }
    
    offsetData.timePointOffsets[sz] = offsetData.timePointOffsets[sz - 1] + fs[sz - 1].points().size();
    
    return offsetData;
  }
  
  template <typename T>
  __device__
  float l1_inner_prod(T, T, T t, T b)
  {
    return t * b;
  }
  
  __device__
  float l1_inner_prod_f32_impl(float l, float r, float t, float b)
  {
    return l1_inner_prod<float>(l, r, t, b);
  }
  
  __device__
  double l1_inner_prod_f64_impl(double l, double r, double t, double b)
  {
    return l1_inner_prod<double>(l, r, t, b);
  }
  
  // Return either the user maximum GPU count, or the number of physical GPUs present (whichever is smaller)
  int get_gpu_limit()
  {
    return 1;
    int nGpus;
    CHK_CUDA(cudaGetDeviceCount(&nGpus));
    // TODO: user limit
    return nGpus;
  }
  
  // Return the maximum number of T's that can be allocated on a single GPU
  template <typename T>
  size_t get_max_allocation_n(int nGpus) // Use first nGpus GPUs
  {
    constexpr float allocationPct = 0.8f; // Use at most this percentage of free GPU ram for matrix (leave some space for other stuff)
    
    size_t retVal = std::numeric_limits<size_t>::max();
    
    for (auto i = 0; i < nGpus; ++i)
    {
      CHK_CUDA(cudaSetDevice(i));
      
      size_t freeMem;
      size_t totalMem;
      CHK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
      
      size_t maxMatrixAllocSz = static_cast<float>(freeMem) * allocationPct;
      size_t maxAllocationN = maxMatrixAllocSz / sizeof(float);
      maxAllocationN = (maxAllocationN / 1024) * 1024;
      
      if (verbose)
      {
        std::cout << "GPU " << i << " has allocation limit " << maxAllocationN << std::endl;
      }
      
      retVal = std::min(retVal, maxAllocationN);
    }
    
    return retVal;
  }
  
  template <typename T>
  size_t get_row_size(int nGpus, size_t nPcfs)
  {
    constexpr size_t nExtraSplits = 2; // Extra subdivisions to give the scheduler something to work with
    
    auto maxAllocationN = get_max_allocation_n<T>(nGpus);
    auto maxRowSz = maxAllocationN / nPcfs;
    
    maxRowSz = std::min(maxRowSz, nPcfs);
    maxRowSz /= nGpus;
    maxRowSz /= nExtraSplits;
    maxRowSz = std::max(maxRowSz, 1ul);
    
    // TODO: detect too low # pcfs -> run everything on one GPU
    
    return maxRowSz;
  }
  
  template <typename T>
  std::vector<std::pair<size_t, size_t>> get_block_row_boundaries(size_t rowSz, size_t nPcfs)
  {
    std::vector<std::pair<size_t, size_t>> boundaries;
    for (size_t i = 0ul;; i += rowSz)
    {
      boundaries.emplace_back(i, std::min(i + rowSz - 1ul, nPcfs));
      if (boundaries.back().second >= nPcfs)
      {
        boundaries.back().second = nPcfs - 1;
        return boundaries;
      }
    }
  }
  
  template <typename Tt, typename Tv>
  DeviceStorage<Tt, Tv>
  make_device_storage(size_t rowSz, size_t nPcfs, const HostPcfOffsetData<Tt, Tv>& hostOffsetData)
  {
    DeviceStorage<Tt, Tv> storage;
    
    storage.matrix = mpcf::CudaDeviceArray<Tv>(rowSz * nPcfs);
    storage.points = mpcf::CudaDeviceArray<SimplePoint<Tt, Tv>>(hostOffsetData.points);
    storage.timePointOffsets = mpcf::CudaDeviceArray<size_t>(hostOffsetData.timePointOffsets);
    
    return storage;
  }
  
  template <typename Tt, typename Tv>
  std::vector<DeviceStorage<Tt, Tv>>
  make_device_storages(int nGpus, size_t rowSz, size_t nPcfs, const HostPcfOffsetData<Tt, Tv>& hostOffsetData)
  {
    std::vector<DeviceStorage<Tt, Tv>> storages;
    storages.resize(nGpus);
    
    for (auto iGpu = 0; iGpu < nGpus; ++iGpu)
    {
      CHK_CUDA(cudaSetDevice(iGpu));
      storages[iGpu] = make_device_storage(rowSz, nPcfs, hostOffsetData);
    }
    
    return storages;
  }
  
  template <typename Tt, typename Tv>
  void
  copy_offset_data_to_active_device(int iGpu, IntegrationContext<Tt, Tv>& ctx)
  {
    ctx.deviceStorages[iGpu].points.toDevice(ctx.hostOffsetData.points);
    ctx.deviceStorages[iGpu].timePointOffsets.toDevice(ctx.hostOffsetData.timePointOffsets);
  }
  
  template <typename Tt, typename Tv>
  void
  copy_offset_data_to_devices(IntegrationContext<Tt, Tv>& ctx)
  {
    for (auto iGpu = 0; iGpu < ctx.nGpus; ++iGpu)
    {
      CHK_CUDA(cudaSetDevice(iGpu));
      copy_offset_data_to_active_device(iGpu, ctx);
    }
  }
  
  template <typename Tt, typename Tv>
  IntegrationContext<Tt, Tv>
  make_context(Tv* out, const std::vector<mpcf::Pcf<Tt, Tv>>& fs, mpcf::DeviceOp<Tt, Tv> op)
  {
    IntegrationContext<Tt, Tv> ctx;
    
    ctx.nPcfs = fs.size();
    ctx.nGpus = get_gpu_limit();
    auto rowSz = get_row_size<Tv>(ctx.nGpus, ctx.nPcfs);
    
    ctx.blockRowBoundaries = get_block_row_boundaries<Tv>(rowSz, ctx.nPcfs);
    ctx.hostOffsetData = get_host_offset_data<Tt, Tv>(fs);
    ctx.deviceStorages = make_device_storages<Tt, Tv>(ctx.nGpus, rowSz, ctx.nPcfs, ctx.hostOffsetData);
    
    copy_offset_data_to_devices(ctx);
    
    ctx.hostMatrix = out;
    
    ctx.blockDim = dim3(128,1,1);
    
    return ctx;
  }
  
  template <typename Tt, typename Tv>
  void
  exec_gpu(int iRow, const tf::Executor& executor, IntegrationContext<Tt, Tv>& ctx)
  {
    
    
    auto iGpu = executor.this_worker_id(); // Worker IDs are guaranteed to be 0...(n-1) for n threads.
    
    std::cout << "Exec row " << iRow << " on GPU " << iGpu << std::endl;
    
    CHK_CUDA(cudaSetDevice(iGpu));
    
    ctx.deviceStorages[iGpu].matrix.clear();
    
    Tv* hostMatrix = ctx.hostMatrix;
    Tv* deviceMatrix = ctx.deviceStorages[iGpu].matrix.get();
    
    auto rowStart = ctx.blockRowBoundaries[iRow].first;
    auto rowEnd = ctx.blockRowBoundaries[iRow].second;
    
    auto start = rowStart * ctx.nPcfs;
    auto end = rowEnd * ctx.nPcfs + ctx.nPcfs + 1;
    
    std::cout << "Block row " << rowStart << " to " << rowEnd << " matrix start " << start << " matrix end " << end << std::endl;
    
  }
  
  template <typename Tt, typename Tv>
  void
  schedule_block_rows(tf::Executor& executor, IntegrationContext<Tt, Tv>& ctx)
  {
    for (auto i = 0ul; i < ctx.blockRowBoundaries.size(); ++i)
    {
      executor.silent_async([&ctx, &executor, i]{ exec_gpu(i, executor, ctx); });
    }
  }
  
  template <typename Tt, typename Tv>
  void
  cuda_matrix_integrate_impl(Tv* out, const std::vector<mpcf::Pcf<Tt, Tv>>& fs, mpcf::DeviceOp<Tt, Tv> op)
  {
    auto ctx = make_context(out, fs, op);
    //params.timePointOffsets = &
    
    std::memset(out, 0, fs.size() * fs.size() * sizeof(Tv));
    
    tf::Executor hostWorkers(ctx.nGpus); // One worker thread per GPU
    
    schedule_block_rows<Tt, Tv>(hostWorkers, ctx);
    
    hostWorkers.wait_for_all();
  }
}

mpcf::DeviceOp<float, float>
mpcf::device_ops::l1_inner_prod_f32()
{
  return &l1_inner_prod_f32_impl;
}

mpcf::DeviceOp<double, double>
mpcf::device_ops::l1_inner_prod_f64()
{
  return &l1_inner_prod_f64_impl;
}

void
mpcf::detail::cuda_matrix_integrate_f32(float* out, const std::vector<Pcf_f32>& fs, DeviceOp<float, float> op)
{
  cuda_matrix_integrate_impl<float, float>(out, fs, op);
}

void
mpcf::detail::cuda_matrix_integrate_f64(double* out, const std::vector<Pcf_f64>& fs, DeviceOp<double, double> op)
{
  cuda_matrix_integrate_impl<double, double>(out, fs, op);
}
