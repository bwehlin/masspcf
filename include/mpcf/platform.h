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

#ifndef MPCF_PLATFORM_H
#define MPCF_PLATFORM_H

  #ifndef BUILD_WITH_CUDA

    #ifndef __host__
      #define __host__
    #endif

    #ifndef __device__
      #define __device__
    #endif

  #endif

  #ifdef _WIN32
    #define MPCF_EXPORT_API __declspec(dllexport)
  #else
    #define MPCF_EXPORT_API
  #endif

#endif
