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

#ifndef MASSPCF_IO_H
#define MASSPCF_IO_H

#include "io/io_stream.h"
#include "io/point_io.h"
#include "io/pcf_io.h"
#include "io/tensor_io.h"
#include "tensor.h"
#include "version.h"

#include "pcf.h"
#include "point_cloud.h"
#include "persistence/barcode.h"

#include <variant>
#include <cstdint>
#include <bit>

#ifdef __APPLE__
#include "TargetConditionals.h"
#endif

namespace mpcf
{
  namespace io::detail
  {
    constexpr const int FormatVersion = 1;

    inline void write_endianness(std::ostream& os)
    {
      if constexpr (std::endian::native == std::endian::little)
      {
        write_binary_string(os, "e");
      }
      else if constexpr (std::endian::native == std::endian::big)
      {
        write_binary_string(os, "E");
      }
      else
      {
        throw std::runtime_error("System not supported (unknown endianness).");
      }
    }

    inline void write_format_version(std::ostream& os)
    {
      write_binary_record(os, std::to_string(FormatVersion));
    }



    inline void write_platform(std::ostream& os)
    {
      // This is just for record keeping at the moment. We don't really use the information. The main purpose is if we
      // get a bug, having this information could help us track down the problem if it's related to a specific platform.

      std::string system =
#if defined(_WIN32) || defined(_WIN64)
      "windows";
#elif defined(__linux__)
      "linux";
#elif __APPLE__
#if defined(TARGET_OS_MAC)
      "osx";
#elif defined(TARGET_OS_IPHONE) or defined(TARGET_IPHONE_SIMULATOR)
      "ios";
#else
      "apple_other";
#endif
#elif defined(ANDROID)
      "android";
#else
      "other";
#endif

      std::string arch =
#if defined(__x86_64__) || defined(_M_X64)
      "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
      "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
      "arm64";
#elif defined(__arm__) || defined(_M_ARM)
      "arm";
#elif defined(__riscv) && (__riscv_xlen == 64)
      "riscv64";
#else
      "unknown-arch";
#endif

      write_string(os, system + "-" + arch);
    }

    inline void write_header(std::ostream& os)
    {
      write_binary_string(os, "\1MPCF");
      write_endianness(os); // Will likely be little-endian until the end of time, but just to be sure!
      write_format_version(os);
      write_binary_record(os, PROJECT_VERSION_FULL);
      write_binary_record(os, PROJECT_BUILD_DATE);
    }

  }

  template <IsTensor TensorT>
  void write(const TensorT& tensor, std::ostream& os)
  {
    io::detail::write_header(os);
    io::detail::write_binary_record(os, "t"); // "t" is single tensor
    io::detail::write_tensor(os, tensor);
  }




}

#endif //MASSPCF_IO_H
