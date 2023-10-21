#ifndef MPCF_CUDA_MATRIX_INTEGRATE_STRUCTS
#define MPCF_CUDA_MATRIX_INTEGRATE_STRUCTS

#include "cuda_device_array.cuh"
#include "cuda_functional_support.cuh"

namespace mpcf::internal
{
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
      //mpcf::DeviceOp<Tt, Tv>* op;
    };
    
    Tv* hostMatrix;
    
    std::vector<DeviceStorage<Tt, Tv>> deviceStorages;
    HostPcfOffsetData<Tt, Tv> hostOffsetData;
    std::vector<std::pair<size_t, size_t>> blockRowBoundaries;
    
    //mpcf::DeviceOp<Tt, Tv>* op;
    
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
      
      return params;
    }
    
  };
}

#endif