#include "algorithms/cuda_matrix_integrate.h"

#include "point.h"
#include "cuda_util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <utility>
#include <vector>

namespace
{
  bool verbose = true;
  
  template <typename Tt, typename Tv>
  struct CpuPcfOffsetData
  {
    std::vector<size_t> nTimePointOffsets;
    std::vector<mpcf::Point<Tt, Tv>> points;
  };
  
  template <typename Tt, typename Tv>
  struct GpuPcfOffsetData
  {
    size_t* nTimePointOffsets = nullptr;
    mpcf::Point<Tt, Tv>* points = nullptr;
    std::size_t n;
  };
  
  template <typename Tt, typename Tv>
  CpuPcfOffsetData<Tt, Tv>
  get_cpu_offset_data(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
  {
    CpuPcfOffsetData<Tt, Tv> offsetData;
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
        offsetData.points[coffs + j] = f[j];
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
  void
  cuda_matrix_integrate_impl(Tv* out, const std::vector<mpcf::Pcf<Tt, Tv>>& fs, mpcf::DeviceOp<Tt, Tv> op)
  {
    auto nGpus = get_gpu_limit();
    auto rowSz = get_row_size<Tv>(nGpus, fs.size());
    
    if (verbose)
    {
      std::cout << "Row size: " << rowSz << std::endl;
    }
    
    auto blockRowBoundaries = get_block_row_boundaries<Tv>(rowSz, fs.size());
    if (verbose)
    {
      for (auto const & bdry : blockRowBoundaries)
      {
        std::cout << "Row [" << bdry.first << ", " << bdry.second << "]" << std::endl;
      }
    }
    
    auto cpuOffsetData = get_cpu_offset_data<Tt, Tv>(fs);
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
