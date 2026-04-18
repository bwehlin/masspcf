/*
* Copyright 2024-2026 Bjorn Wehlin
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


#ifndef MPCF_CUDA_UTIL_H
#define MPCF_CUDA_UTIL_H

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

namespace mpcf
{
  // Thrown whenever a CUDA runtime call returns a non-success status.
  // Derives from std::runtime_error so existing catch blocks keep working.
  // Callers that want to single out OOM compare code() to
  // cudaErrorMemoryAllocation directly.
  class cuda_error : public std::runtime_error
  {
  public:
    cuda_error(const char* file, int line, cudaError_t code)
      : std::runtime_error(build_what(file, line, code)), m_code(code) { }

    [[nodiscard]] cudaError_t code() const noexcept { return m_code; }

  private:
    static std::string build_what(const char* file, int line, cudaError_t code)
    {
      return std::string(file) + ":" + std::to_string(line) + ": " + cudaGetErrorString(code);
    }

    cudaError_t m_code;
  };
}

#define CHK_CUDA(x) do { \
  cudaError_t _mpcf_chk_rv = (x); \
  if (_mpcf_chk_rv != cudaSuccess) { \
    throw ::mpcf::cuda_error(__FILE__, __LINE__, _mpcf_chk_rv); \
  } \
} while (0)

#endif
