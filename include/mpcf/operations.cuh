/*
* Copyright 2024-2025 Bjorn Wehlin
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
    
    __host__ __device__ Tv operator()(Tv x) const
    {
      return x;
    }
  };
  
  template <typename Tt, typename Tv>
  struct OperationLpDist
  {
    Tv p = 2.0;
    OperationLpDist() = default;
    OperationLpDist(Tv pv)
      : p(pv)
    { }
    
    __host__ __device__ Tv operator()(Tv t, Tv b) const
    {
      return pow(abs(t - b), p);
    }
    
    __host__ __device__ Tv operator()(Tv x) const
    {
      return pow(x, Tv(1) / p);
    }
  };
  
  template <typename Tt, typename Tv>
  struct OperationL2InnerProduct
  {
    __host__ __device__ Tv operator()(Tv t, Tv b) const
    {
      return t * b;
    }
    
    __host__ __device__ Tv operator()(Tv x) const
    {
      return x;
    }
  };
}

#endif
