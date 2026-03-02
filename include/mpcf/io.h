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

#include "io_stream.h"
#include "tensor.h"
#include "version.h"

#include "pcf.h"
#include "point_cloud.h"
#include "persistence/barcode.h"

#include <variant>
#include <cstdint>
#include <bit>

namespace mpcf
{
  using StreamableTensor = std::variant<
      Tensor<float>,
      Tensor<double>,

      Tensor<Pcf<float, float>>,
      Tensor<Pcf<double, double>>,

      Tensor<PointCloud<float>>,
      Tensor<PointCloud<double>>,

      Tensor<ph::Barcode<float>>,
      Tensor<ph::Barcode<double>>
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
      return std::visit([](auto&& arg) -> TensorFormat {
        using T = Tensor<std::decay_t<decltype(arg)>>;

        if      constexpr (std::is_same_v<T, float>)      { return TensorFormat{ .baseFormat = 1, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, double>)     { return TensorFormat{ .baseFormat = 1, .subFormat = 64 }; }

        else if constexpr (std::is_same_v<T, Pcf<float, float>>)   { return TensorFormat{ .baseFormat = 100, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, Pcf<double, double>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 64 }; }

        else if constexpr (std::is_same_v<T, PointCloud<float>>)    { return TensorFormat{ .baseFormat = 1000, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, PointCloud<double>>)   { return TensorFormat{ .baseFormat = 1000, .subFormat = 64 }; }

        else if constexpr (std::is_same_v<T, ph::Barcode<float>>)   { return TensorFormat{ .baseFormat = 2000, .subFormat = 32 }; }
        else if constexpr (std::is_same_v<T, ph::Barcode<double>>)  { return TensorFormat{ .baseFormat = 2000, .subFormat = 64 }; }

        throw std::runtime_error("Tensor type not supported.");
      }, tensor);
    }

    inline void assert_not_bad(std::ostream& os)
    {
      if (os.bad())
      {
        throw std::runtime_error("Bad stream.");
      }
    }

    inline void write_binary_string(std::ostream& os, const std::string& str)
    {
      os.write(str.c_str(), str.size());
      assert_not_bad(os);
    }

    inline void write_binary_record(std::ostream& os, const std::string& str)
    {
      write_binary_string(os, str);
      write_binary_string(os, "\36"); // \36 is ASCII record separator (decimal 30)
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

    inline void write_platform(std::string& os)
    {
#ifdef WIN32
      write_binary_record(os, "WIN32");
#elif __linux__
      //write_binary_record(os, "LINUX");
#endif
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

  }


}

#endif //MASSPCF_IO_H
