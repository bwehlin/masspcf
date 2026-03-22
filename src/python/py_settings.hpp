// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Created by bjorn on 3/1/2026.
//

#ifndef MASSPCF_PY_SETTINGS_H
#define MASSPCF_PY_SETTINGS_H

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace mpcf_py
{
  struct Settings
  {
    bool forceCpu = false; // Force computation on CPU
    size_t cudaThreshold = 500; // Number of pcfs required for CUDA run to be invoked over CPU
    bool deviceVerbose = false; // Print message for which device (CPU/CUDA) is used for the computation

#ifdef BUILD_WITH_CUDA
    dim3 blockDim = dim3(1, 32, 1);
#endif

  };

  extern Settings g_settings;
}

#endif //MASSPCF_PY_SETTINGS_H