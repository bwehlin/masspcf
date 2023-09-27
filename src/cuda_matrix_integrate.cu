#include "algorithms/cuda_matrix_integrate.h"

#include "point.h"
#include "cuda_util.h"
#include "cuda_device_array.h"
#include "block_matrix_support.h"

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
    struct DeviceKernelParams
    {
      Tv* matrix;
      size_t* timePointOffsets;
      SimplePoint<Tt, Tv>* points;
      size_t nPcfs;
      mpcf::DeviceOp<Tt, Tv> op;
    };
    
    Tv* hostMatrix;
    
    std::vector<DeviceStorage<Tt, Tv>> deviceStorages;
    HostPcfOffsetData<Tt, Tv> hostOffsetData;
    std::vector<std::pair<size_t, size_t>> blockRowBoundaries;
    
    mpcf::DeviceOp<Tt, Tv> op;
    
    size_t nPcfs;
    
    int nGpus;
    dim3 blockDim;
    
    DeviceKernelParams make_kernel_params(int iGpu) const
    {
      DeviceKernelParams params;
      auto & storage = deviceStorages[iGpu];
      
      params.matrix = storage.matrix.get();
      params.points = storage.points.get();
      params.timePointOffsets = storage.timePointOffsets.get();
      
      params.nPcfs = nPcfs;
      params.op = op;
      
      return params;
    }
    
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
  std::vector<std::pair<size_t, size_t>> get_block_row_boundaries(int nGpus, size_t nPcfs)
  {
    auto maxAllocationN = get_max_allocation_n<T>(nGpus);
    auto nSplits = nGpus * 2; // Give the scheduler something to work with
    auto rowHeight = mpcf::internal::get_row_size(maxAllocationN, nSplits, nPcfs);
    return mpcf::internal::get_block_row_boundaries(rowHeight, nPcfs);
  }
  
  template <typename Tt, typename Tv>
  DeviceStorage<Tt, Tv>
  make_device_storage(size_t rowHeight, size_t nPcfs, const HostPcfOffsetData<Tt, Tv>& hostOffsetData)
  {
    DeviceStorage<Tt, Tv> storage;
    
    storage.matrix = mpcf::CudaDeviceArray<Tv>(rowHeight * nPcfs);
    storage.points = mpcf::CudaDeviceArray<SimplePoint<Tt, Tv>>(hostOffsetData.points);
    storage.timePointOffsets = mpcf::CudaDeviceArray<size_t>(hostOffsetData.timePointOffsets);
    
    return storage;
  }
  
  template <typename Tt, typename Tv>
  void
  init_device_storages(IntegrationContext<Tt, Tv>& ctx)
  {
    auto & storages = ctx.deviceStorages;
    storages.resize(ctx.nGpus);
    
    auto maxRowHeight = ctx.blockRowBoundaries[0].second + 1;
    
    for (auto iGpu = 0; iGpu < ctx.nGpus; ++iGpu)
    {
      CHK_CUDA(cudaSetDevice(iGpu));
      storages[iGpu] = make_device_storage(maxRowHeight, ctx.nPcfs, ctx.hostOffsetData);
    }
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
    
    ctx.blockRowBoundaries = get_block_row_boundaries<Tv>(ctx.nGpus, ctx.nPcfs);
    ctx.hostOffsetData = get_host_offset_data<Tt, Tv>(fs);
    init_device_storages<Tt, Tv>(ctx);
    
    copy_offset_data_to_devices(ctx);
    
    ctx.hostMatrix = out;
    
    ctx.blockDim = dim3(1,8,1);
    
    return ctx;
  }
  
  struct RowInfo
  {
    size_t rowStart;
    size_t rowHeight;
    size_t iRow;
  };
  
  template <typename Tt, typename Tv>
  __global__
  void cuda_iterate_rectangles(
      typename IntegrationContext<Tt, Tv>::DeviceKernelParams params, 
      RowInfo rowInfo)
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
    
    auto t = 0.f; // TODO: a
    auto tPrev = t;
    
    while (true /* t < b */)
    {
      tPrev = t;
      return;
    }
    
    
    params.matrix[iBlock * params.nPcfs + j] = i + j + 50;
  }
  
  template <typename Tt, typename Tv>
  __device__
  void cuda_integrate(typename IntegrationContext<Tt, Tv>::DeviceKernelParams params, size_t rowStart, size_t rowHeight, size_t iRow)
  {
    
  }
  
  template <typename Tt, typename Tv>
  void
  exec_gpu(size_t iRow, const tf::Executor& executor, IntegrationContext<Tt, Tv>& ctx)
  {
    auto iGpu = executor.this_worker_id(); // Worker IDs are guaranteed to be 0...(n-1) for n threads.
    
    std::cout << "Exec row " << iRow << " on GPU " << iGpu << std::endl;
    
    CHK_CUDA(cudaSetDevice(iGpu));
    
    ctx.deviceStorages[iGpu].matrix.clear();
    
    Tv* hostMatrix = ctx.hostMatrix;
    Tv* deviceMatrix = ctx.deviceStorages[iGpu].matrix.get();
    
    auto const & rowBoundaries = ctx.blockRowBoundaries[iRow];
    
    auto rowStart = rowBoundaries.first;
    auto rowEnd = rowBoundaries.second;
    
    auto start = rowStart * ctx.nPcfs;
    auto end = rowEnd * ctx.nPcfs + ctx.nPcfs + 1;
    
    auto rowHeight = mpcf::internal::get_row_height_from_boundaries(rowBoundaries);
    auto gridDims = mpcf::internal::get_grid_dims(ctx.blockDim, rowHeight, ctx.nPcfs);
    
    auto params = ctx.make_kernel_params(iGpu);
    
    RowInfo rowInfo;
    rowInfo.rowHeight = rowHeight;
    rowInfo.rowStart = rowStart;
    rowInfo.iRow = iRow;
    
    cuda_iterate_rectangles<Tt, Tv><<<gridDims, ctx.blockDim>>>(params, rowInfo);
    CHK_CUDA(cudaPeekAtLastError());
    
    auto* target = &hostMatrix[rowStart * ctx.nPcfs];
    auto nEntries = rowHeight * ctx.nPcfs;

    ctx.deviceStorages[iGpu].matrix.toHost(target, nEntries);
    
    //CHK_CUDA(cudaDeviceSynchronize()); 
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

template <>
void 
mpcf::cuda_matrix_l1_dist<float, float>(float* out, const std::vector<Pcf<float, float>>& fs)
{
  
}

template <>
void 
mpcf::cuda_matrix_l1_dist<double, double>(double* out, const std::vector<Pcf<double, double>>& fs)
{
  
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
