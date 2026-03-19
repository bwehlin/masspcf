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
#include "tensor.h"
#include "version.h"

#include "functional/pcf.h"
#include "point_cloud.h"
#include "symmetric_matrix.h"
#include "persistence/barcode.h"

#include <variant>
#include <cstdint>
#include <bit>
#include <string_view>

#ifdef __APPLE__
#include "TargetConditionals.h"
#endif

namespace mpcf
{
  namespace io::detail
  {
    constexpr const std::string_view HeaderIdBytes = "\1MPCF"; // This should never change!

    // This should change as soon as an older version would not be able to read the data produced by the current version.
    constexpr const int FormatVersion = 2;

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

    inline void read_endianness(std::istream& is)
    {
      auto endiannessString = read_binary_string(is, 1);
      if constexpr (std::endian::native == std::endian::little)
      {
        if (endiannessString == "e")
        {
          return;
        }
      }
      else if constexpr (std::endian::native == std::endian::big)
      {
        if (endiannessString == "E")
        {
          return;
        }
      }
      else
      {
        throw std::runtime_error("System not supported (unknown endianness).");
      }
      throw std::runtime_error("Data were saved on a platform with different endianness and cannot be read on this platform. If you need to move data in this way, please file an issue.");
    }

    inline void write_format_version(std::ostream& os)
    {
      write_bytes<int>(os, FormatVersion);
    }

    inline int read_format_version(std::istream& is)
    {
      return read_bytes<int>(is);
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
      write_binary_string(os, HeaderIdBytes);
      write_endianness(os); // Will likely be little-endian until the end of time, but just to be sure!
      write_format_version(os);
      write_string(os, PROJECT_VERSION_FULL);
      write_string(os, PROJECT_BUILD_DATE);
      write_platform(os); // Added in format version 2
    }

    inline void read_header(std::istream& is)
    {
      auto idBytes = read_binary_string(is, HeaderIdBytes.length());
      if (idBytes != HeaderIdBytes)
      {
        throw std::runtime_error("Unrecognized file format.");
      }

      read_endianness(is);

      auto formatVersion = read_format_version(is);
      if (formatVersion < 1 || formatVersion > FormatVersion)
      {
        throw std::runtime_error("Input file has format version " + std::to_string(formatVersion) + ". This version of masspcf reads format versions 1 through " + std::to_string(FormatVersion) + ".");
      }

      read_string(is); // PROJECT_VERSION_FULL
      read_string(is); // PROJECT_BUILD_DATE

      if (formatVersion >= 2)
      {
        read_string(is); // platform (added in format version 2)
      }
    }

  }

  enum class FormatType : uint32_t
  {
    SingleTensor = 1
  };

  inline std::string formatName(uint32_t tp)
  {
    // C++26 reflection, please!
    if (tp == static_cast<uint32_t>(FormatType::SingleTensor))
    {
      return "SingleTensor";
    }
    else
    {
      return "Unknown(" + std::to_string(tp) + ")";
    }
  }

  template <IsTensor TensorT>
  void write(const TensorT& tensor, std::ostream& os)
  {
    io::detail::write_header(os);
    io::detail::write_bytes<uint32_t>(os, static_cast<uint32_t>(FormatType::SingleTensor)); // "t" is single tensor
    io::detail::write_tensor(os, tensor);
  }

  template <IsTensor TensorT>
  TensorT read(std::istream& is)
  {
    io::detail::read_header(is);

    auto formatType = io::detail::read_bytes<uint32_t>(is);
    if (formatType != static_cast<uint32_t>(FormatType::SingleTensor))
    {
      throw std::runtime_error("Expected format type " + formatName(static_cast<uint32_t>(FormatType::SingleTensor))
          + " for this operation but got format type " + formatName(formatType));
    }

    io::detail::TensorFormat format;

    format.baseFormat = io::detail::read_bytes<std::int32_t>(is);
    format.subFormat = io::detail::read_bytes<std::int32_t>(is);

    auto expectedFormat = io::detail::tensorFormat<typename TensorT::value_type>();
    if (format != expectedFormat)
    {
      throw std::runtime_error("Unexpected tensor format " + format.toString() + " where " + expectedFormat.toString() + " was expected.");
    }

    return io::detail::read_tensor<typename TensorT::value_type>(is);
  }

  inline io::detail::StreamableTensor read_any_tensor(std::istream& is)
  {
    io::detail::read_header(is);

    auto formatType = io::detail::read_bytes<uint32_t>(is);
    if (formatType != static_cast<uint32_t>(FormatType::SingleTensor))
    {
      throw std::runtime_error("Expected format type " + formatName(static_cast<uint32_t>(FormatType::SingleTensor))
                               + " for this operation but got format type " + formatName(formatType));
    }

    auto format = io::detail::read_tensor_format(is);

    if      (format == io::detail::tensorFormat<float32_t>()) { return io::detail::read_tensor<float32_t>(is); }
    else if (format == io::detail::tensorFormat<float64_t>()) { return io::detail::read_tensor<float64_t>(is); }

    else if (format == io::detail::tensorFormat<Pcf<float32_t, float32_t>>()) { return io::detail::read_tensor<Pcf<float32_t, float32_t>>(is); }
    else if (format == io::detail::tensorFormat<Pcf<float64_t, float64_t>>()) { return io::detail::read_tensor<Pcf<float64_t, float64_t>>(is); }

    else if (format == io::detail::tensorFormat<Pcf<int32_t, int32_t>>()) { return io::detail::read_tensor<Pcf<int32_t, int32_t>>(is); }
    else if (format == io::detail::tensorFormat<Pcf<int64_t, int64_t>>()) { return io::detail::read_tensor<Pcf<int64_t, int64_t>>(is); }

    else if (format == io::detail::tensorFormat<PointCloud<float32_t>>()) { return io::detail::read_tensor<PointCloud<float32_t>>(is); }
    else if (format == io::detail::tensorFormat<PointCloud<float64_t>>()) { return io::detail::read_tensor<PointCloud<float64_t>>(is); }

    else if (format == io::detail::tensorFormat<SymmetricMatrix<float32_t>>()) { return io::detail::read_tensor<SymmetricMatrix<float32_t>>(is); }
    else if (format == io::detail::tensorFormat<SymmetricMatrix<float64_t>>()) { return io::detail::read_tensor<SymmetricMatrix<float64_t>>(is); }

    else if (format == io::detail::tensorFormat<ph::Barcode<float32_t>>()) { return io::detail::read_tensor<ph::Barcode<float32_t>>(is); }
    else if (format == io::detail::tensorFormat<ph::Barcode<float64_t>>()) { return io::detail::read_tensor<ph::Barcode<float64_t>>(is); }

    else
    {
      throw std::runtime_error("Unhandled tensor type (" + std::to_string(format.baseFormat) + ", " + std::to_string(format.subFormat) + ")");
    }

  }

}

#endif //MASSPCF_IO_H
