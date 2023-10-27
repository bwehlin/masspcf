#ifndef MPCF_OPERATIONS_CUH
#define MPCF_OPERATIONS_CUH

#ifdef BUILD_WITH_CUDA
  #include <cuda_runtime.h>
#else
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

namespace mpcf
{
  template <typename Tt, typename Tv>
  struct OperationL1Dist
  {
    __host__ __device__ Tv operator()(Tv t, Tv b) const
    {
      return abs(t - b);
    }
  };
  
  template <typename Tt, typename Tv>
  struct LpDist
  {
    Tv p = 2.0;
    __host__ __device__ Tv operator()(Tt l, Tt r, Tv t, Tv b) const
    {
      return pow(abs(t - b), p);
    }
  };
}

#endif
