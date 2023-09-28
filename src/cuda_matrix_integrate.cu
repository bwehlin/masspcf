#include "algorithms/cuda_matrix_integrate.h"

#include "point.h"
#include "cuda_util.h"
#include "cuda_device_array.h"
#include "block_matrix_support.h"
#include "cuda_functional_support.h"
#include "cuda_matrix_integrate_structs.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <utility>
#include <vector>

#include <taskflow/taskflow.hpp>

namespace
{
  bool verbose = true;
  
  template <typename Tt, typename Tv>
  mpcf::internal::HostPcfOffsetData<Tt, Tv>
  get_host_offset_data(const std::vector<mpcf::Pcf<Tt, Tv>>& fs)
  {
    mpcf::internal::HostPcfOffsetData<Tt, Tv> offsetData;
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
    //return 1;
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
  mpcf::internal::DeviceStorage<Tt, Tv>
  make_device_storage(size_t rowHeight, size_t nPcfs, const mpcf::internal::HostPcfOffsetData<Tt, Tv>& hostOffsetData)
  {
    mpcf::internal::DeviceStorage<Tt, Tv> storage;
    
    storage.matrix = mpcf::CudaDeviceArray<Tv>(rowHeight * nPcfs);
    storage.points = mpcf::CudaDeviceArray<mpcf::internal::SimplePoint<Tt, Tv>>(hostOffsetData.points);
    storage.timePointOffsets = mpcf::CudaDeviceArray<size_t>(hostOffsetData.timePointOffsets);
    
    return storage;
  }
  
  template <typename Tt, typename Tv>
  void
  init_device_storages(mpcf::internal::IntegrationContext<Tt, Tv>& ctx)
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
  copy_offset_data_to_active_device(int iGpu, mpcf::internal::IntegrationContext<Tt, Tv>& ctx)
  {
    ctx.deviceStorages[iGpu].points.toDevice(ctx.hostOffsetData.points);
    ctx.deviceStorages[iGpu].timePointOffsets.toDevice(ctx.hostOffsetData.timePointOffsets);
  }
  
  template <typename Tt, typename Tv>
  void
  copy_offset_data_to_devices(mpcf::internal::IntegrationContext<Tt, Tv>& ctx)
  {
    for (auto iGpu = 0; iGpu < ctx.nGpus; ++iGpu)
    {
      CHK_CUDA(cudaSetDevice(iGpu));
      copy_offset_data_to_active_device(iGpu, ctx);
    }
  }
  
  template <typename Tt, typename Tv>
  mpcf::internal::IntegrationContext<Tt, Tv>
  make_context(Tv* out, const std::vector<mpcf::Pcf<Tt, Tv>>& fs, mpcf::DeviceOp<Tt, Tv>* op)
  {
    mpcf::internal::IntegrationContext<Tt, Tv> ctx;
    
    ctx.nPcfs = fs.size();
    ctx.nGpus = get_gpu_limit();
    
    ctx.blockRowBoundaries = get_block_row_boundaries<Tv>(ctx.nGpus, ctx.nPcfs);
    ctx.hostOffsetData = get_host_offset_data<Tt, Tv>(fs);
    init_device_storages<Tt, Tv>(ctx);
    
    copy_offset_data_to_devices(ctx);
    
    ctx.hostMatrix = out;
    
    ctx.blockDim = dim3(1,8,1);
    
    ctx.op = op;
    
    return ctx;
  }
  
  struct RowInfo
  {
    size_t rowStart;
    size_t rowHeight;
    size_t iRow;
  };
  
  template <typename Tt, typename Tv, typename FOp>
  __device__
  void cuda_iterate_rectangles(
      typename mpcf::internal::IntegrationContext<Tt, Tv>::DeviceKernelParams params, 
      RowInfo rowInfo, size_t fMatrixIdx, size_t gMatrixIdx,
      FOp op)
  {
    Tt t = 0; // TODO: a
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
    
    auto b = 1000; //std::numeric_limits<Tt>::max(); // TODO
    
    while (t < b)
    {
      tprev = t;
      fv = fpts[fi].v;
      gv = gpts[gi].v;

      if (fi + 1 < fsz && gi + 1 < gsz)
      {
        auto delta = fpts[fi+1].t - gpts[gi+1].t;
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
          op(tprev, b, fv, gv);
          return;
        }
      }
      
      t = max(fpts[fi].t, gpts[gi].t);
      op(tprev, t, fv, gv);
    }
    
  }
  
  template <typename Tt, typename Tv>
  __device__
  Tv sub1(Tt, Tt, Tv t, Tv b)
  {
    return abs(t - b);
  }
  
  __device__
  float sub1f(float, float, float t, float b)
  {
    return abs(t - b);
  }
  
  __device__
  double sub1d(double, double, double t, double b)
  {
    return abs(t - b);
  }
  
  template <typename Tt, typename Tv>
  __global__
  void cuda_integrate(
      typename mpcf::internal::IntegrationContext<Tt, Tv>::DeviceKernelParams params, 
      RowInfo rowInfo)
  {
    size_t iBlock = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    
    //printf("Integrate (%d, %d)\n", (int)iBlock, (int)j);
    
    if (iBlock >= rowInfo.rowHeight)
    {
      return;
    }
    
    size_t i = iBlock + rowInfo.rowStart;
    
    if (j < i || i >= params.nPcfs || j >= params.nPcfs)
    {
      return;
    }
    
    auto* op = params.op;
    printf("op %p\n", op);
    
    Tv ret = 0;
    cuda_iterate_rectangles<Tt, Tv>(params, rowInfo, i, j, [&ret, op](Tt l, Tt r, Tv t, Tv b){
      ret += (r - l) * (*op)(l, r, t, b);
    });
    
    //printf("Integrate (%d, %d) -> %f\n", (int)i, (int)j, ret);
    params.matrix[iBlock * params.nPcfs + j] = ret;
  }
  
  template <typename Tt, typename Tv>
  void
  exec_gpu(size_t iRow, const tf::Executor& executor, mpcf::internal::IntegrationContext<Tt, Tv>& ctx)
  {
    auto iGpu = executor.this_worker_id(); // Worker IDs are guaranteed to be 0...(n-1) for n threads.
    
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
    
    mpcf::detail::CudaCallableFunctionPointer<mpcf::DeviceOp<Tt, Tv>> op(ctx.op);
    
    auto params = ctx.make_kernel_params(iGpu);
    params.op = op.ptr;

    RowInfo rowInfo;
    rowInfo.rowHeight = rowHeight;
    rowInfo.rowStart = rowStart;
    rowInfo.iRow = iRow;
    
    cuda_integrate<Tt, Tv><<<gridDims, ctx.blockDim>>>(params, rowInfo);
    CHK_CUDA(cudaPeekAtLastError());
    
    auto* target = &hostMatrix[rowStart * ctx.nPcfs];
    auto nEntries = rowHeight * ctx.nPcfs;

    ctx.deviceStorages[iGpu].matrix.toHost(target, nEntries);
    
    //CHK_CUDA(cudaDeviceSynchronize()); 
  }
  
  template <typename Tt, typename Tv>
  void
  schedule_block_rows(tf::Executor& executor, mpcf::internal::IntegrationContext<Tt, Tv>& ctx)
  {
    for (auto i = 0ul; i < ctx.blockRowBoundaries.size(); ++i)
    {
      executor.silent_async([&ctx, &executor, i]{ exec_gpu(i, executor, ctx); });
    }
  }
  
  template <typename Tt, typename Tv>
  void
  cuda_matrix_integrate_impl(Tv* out, const std::vector<mpcf::Pcf<Tt, Tv>>& fs, mpcf::DeviceOp<Tt, Tv>* op)
  {
    auto ctx = make_context(out, fs, op);
    //params.timePointOffsets = &
    
    std::memset(out, 0, fs.size() * fs.size() * sizeof(Tv));
    
    tf::Executor hostWorkers(ctx.nGpus); // One worker thread per GPU
    
    schedule_block_rows<Tt, Tv>(hostWorkers, ctx);
    
    hostWorkers.wait_for_all();
  }
}

//typedef float (*op_func_f)(float, float, float, float);
//__device__ op_func_f opf = sub1f;
__device__ mpcf::DeviceOp<float, float> opf = sub1f;
//typedef double (*op_func_d)(double, double, double, double);
//__device__ op_func_d opd = sub1d;
__device__ mpcf::DeviceOp<double,double> opd = sub1d;



template <>
void 
mpcf::cuda_matrix_l1_dist<float, float>(float* out, const std::vector<Pcf<float, float>>& fs)
{
  //mpcf::detail::CudaCallableFunctionPointer<op_func_f> f(&opf);
  cuda_matrix_integrate_impl<float, float>(out, fs, &opf);
}

template <>
void 
mpcf::cuda_matrix_l1_dist<double, double>(double* out, const std::vector<Pcf<double, double>>& fs)
{
  //mpcf::detail::CudaCallableFunctionPointer<op_func_d> f(&opd);
  cuda_matrix_integrate_impl<double, double>(out, fs, &opd);
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
mpcf::detail::cuda_matrix_integrate_f32(float* out, const std::vector<Pcf_f32>& fs, DeviceOp<float, float> opa)
{
  //mpcf::detail::CudaCallableFunctionPointer<op_func_f> f(&opf);
  //op_func_f hOpPtr;
  //CHK_CUDA(cudaMemcpyFromSymbol(&hOpPtr, opf, sizeof(op_func_f)));
  cuda_matrix_integrate_impl<float, float>(out, fs, &opf);
}



void
mpcf::detail::cuda_matrix_integrate_f64(double* out, const std::vector<Pcf_f64>& fs, DeviceOp<double, double> opa)
{
  //mpcf::detail::CudaCallableFunctionPointer<op_func_d> f(&opd);
  //op_func_d hOpPtr;
  //CHK_CUDA(cudaMemcpyFromSymbol(&hOpPtr, opd, sizeof(op_func_d))); 
  cuda_matrix_integrate_impl<double, double>(out, fs, &opd);
}
