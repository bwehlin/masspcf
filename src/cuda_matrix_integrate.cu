#include "algorithms/cuda_matrix_integrate.h"

#include "point.h"
#include "cuda_util.h"
#include "cuda_device_array.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <utility>
#include <vector>

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
    std::vector<size_t> nTimePointOffsets;
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
  struct DeviceExecParams
  {
    Tv* hostMatrix;
    
    size_t nPcfs;
    size_t* timePointOffsets;
    SimplePoint<Tt, Tv>* points;
    
    mpcf::DeviceOp<Tt, Tv> op;
    
    dim3 blockDim;
  };
  
  template <typename Tt, typename Tv>
  HostPcfOffsetData<Tt, Tv>
  get_host_offset_data(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
  {
    HostPcfOffsetData<Tt, Tv> offsetData;
    auto sz = fs.size();
    
    offsetData.nTimePointOffsets.resize(sz + 1);
    
    // Compute size required for all PCFs
    auto offset = 0ul;
    for (auto i = 0ul; i < sz; ++i)
    {
      auto const & f = fs[i].points();
      offsetData.nTimePointOffsets[i] = offset;
      offset += f.size();
    }
    
    // Store PCFs
    offsetData.points.resize(offset);
    for (auto i = 0ul; i < sz; ++i)
    {
      auto const & f = fs[i].points();
      auto csz = f.size();
      auto coffs = offsetData.nTimePointOffsets[i];
      for (auto j = 0ul; j < csz; ++j)
      {
        offsetData.points[coffs + j].t = f[j].t;
        offsetData.points[coffs + j].v = f[j].v;
      }
    }
    
    offsetData.nTimePointOffsets[sz] = offsetData.nTimePointOffsets[sz - 1] + fs[sz - 1].points().size();
    
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
    storage.timePointOffsets = mpcf::CudaDeviceArray<size_t>(hostOffsetData.nTimePointOffsets);
    
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
  cuda_matrix_integrate_impl(Tv* out, const std::vector<mpcf::Pcf<Tt, Tv>>& fs, mpcf::DeviceOp<Tt, Tv> op)
  {
    auto nPcfs = fs.size();
    
    auto nGpus = get_gpu_limit();
    auto rowSz = get_row_size<Tv>(nGpus, nPcfs);
    
    if (verbose)
    {
      std::cout << "Row size: " << rowSz << std::endl;
    }
    
    auto blockRowBoundaries = get_block_row_boundaries<Tv>(rowSz, nPcfs);
    if (verbose)
    {
      for (auto const & bdry : blockRowBoundaries)
      {
        std::cout << "Row [" << bdry.first << ", " << bdry.second << "]" << std::endl;
      }
    }
    
    auto hostOffsetData = get_host_offset_data<Tt, Tv>(fs);
    
    DeviceExecParams<Tt, Tv> params;
    params.hostMatrix = nullptr; // Delayed initialization
    //params.timePointOffsets = &
    
    auto deviceStorages = make_device_storages<Tt, Tv>(nGpus, rowSz, nPcfs, hostOffsetData);
    
    
    
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
