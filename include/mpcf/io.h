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

#include "io.h"
#include "io/io_stream.h"
#include "io/point_io.h"
#include "io/pcf_io.h"
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
  using StreamableTensor = std::variant<
      Tensor<float32_t>,
      Tensor<float64_t>,

      Tensor<Pcf<float32_t, float32_t>>,
      Tensor<Pcf<float64_t, float64_t>>,

      Tensor<PointCloud<float32_t>>,
      Tensor<PointCloud<float64_t>>,

      Tensor<ph::Barcode<float32_t>>,
      Tensor<ph::Barcode<float64_t>>
      >;

  namespace io::detail
  {
    constexpr const int FormatVersion = 1;

    struct TensorFormat
    {
      std::int32_t baseFormat;
      std::int32_t subFormat;
    };

    inline TensorFormat getFormat(const StreamableTensor& tensor)
    {
      using namespace std::string_literals;

      return std::visit([](auto&& arg) -> TensorFormat {
        using T = std::decay_t<decltype(arg)>;

        if      constexpr (std::is_same_v<T, float32_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, float64_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 64 }; }

        else if constexpr (std::is_same_v<T, Pcf<float32_t, float32_t>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, Pcf<float64_t, float64_t>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 64 }; }

        else if constexpr (std::is_same_v<T, PointCloud<float32_t>>) { return TensorFormat{ .baseFormat = 1000, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, PointCloud<float64_t>>) { return TensorFormat{ .baseFormat = 1000, .subFormat = 64 }; }

        else if constexpr (std::is_same_v<T, ph::Barcode<float32_t>>) { return TensorFormat{ .baseFormat = 2000, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, ph::Barcode<float64_t>>) { return TensorFormat{ .baseFormat = 2000, .subFormat = 64 }; }

        throw std::runtime_error("Tensor type "s + mpcf::detail::unmangled_typename<T>() +  " not supported.");
      }, tensor);
    }


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

    // For new types, make sure to add io::detail::write_element in their corresponding headers
    template <ArithmeticType T>
    void write_element(std::ostream& os, T elem)
    {
      write_bytes<float64_t>(os, elem);
    }

    template <IsTensor TensorT>
    void write_contiguous_tensor(std::ostream& os, const TensorT& tensor)
    {
      auto format = getFormat(tensor);
      write_bytes<std::int32_t>(os, format.baseFormat);
      write_bytes<std::int32_t>(os, format.subFormat);

      write_bytes<std::uint64_t>(os, tensor.shape().size());
      for (auto i = 0_uz; i < tensor.shape().size(); ++i)
      {
        write_bytes<std::uint64_t>(os, tensor.shape()[i]);
        write_bytes<std::uint64_t>(os, tensor.strides()[i]);
      }

      auto sz = tensor.size();
      for (auto const * elem = tensor.data(); elem != tensor.data() + sz; ++elem)
      {
        write_element(os, *elem);
      }
    }

    template <IsTensor TensorT>
    void write_tensor(std::ostream& os, const TensorT& tensor)
    {
      if (!tensor.is_contiguous())
      {
        auto copy = tensor.copy();
        if (!copy.is_contiguous())
        {
          // To avoid infinite loop
          throw std::runtime_error("Tensor copy is non-contiguous/non-zero-offset (this is a bug, please report it!).");
        }
        write_tensor(os, copy);
        return;
      }
      write_contiguous_tensor(os, tensor);
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
