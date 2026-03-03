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

#ifndef MASSPCF_TENSOR_IO_H
#define MASSPCF_TENSOR_IO_H

#include "io_stream_base.h"
#include "../tensor.h"
#include "../pcf.h"
#include "../persistence/barcode.h"

namespace mpcf::io::detail
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

  struct TensorFormat
  {
    std::int32_t baseFormat;
    std::int32_t subFormat;

    std::string toString() const
    {
      return "(" + std::to_string(baseFormat) + ", " + std::to_string(subFormat) + ")";
    }

    bool operator==(const TensorFormat&) const = default;
    bool operator!=(const TensorFormat&) const = default;
  };

  template <typename U>
  TensorFormat tensorFormat()
  {
    using namespace std::string_literals;
    using T = std::decay_t<U>;

    if      constexpr (std::is_same_v<T, float32_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, float64_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, Pcf<float32_t, float32_t>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, Pcf<float64_t, float64_t>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, PointCloud<float32_t>>) { return TensorFormat{ .baseFormat = 1000, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, PointCloud<float64_t>>) { return TensorFormat{ .baseFormat = 1000, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, ph::Barcode<float32_t>>) { return TensorFormat{ .baseFormat = 2000, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, ph::Barcode<float64_t>>) { return TensorFormat{ .baseFormat = 2000, .subFormat = 64 }; }

    throw std::runtime_error("Tensor type "s + mpcf::detail::unmangled_typename<T>() +  " not supported.");
  }

  inline TensorFormat getTensorFormat(const StreamableTensor& tensor)
  {
    return std::visit([](auto&& arg) -> TensorFormat {
      using TensorT = std::decay_t<decltype(arg)>;
      using T = TensorT::value_type;
      return tensorFormat<T>();
    }, tensor);
  }

  template <IsTensor TensorT>
    void write_contiguous_tensor(std::ostream& os, const TensorT& tensor)
  {
    auto format = getTensorFormat(tensor);
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
      write_element<typename TensorT::value_type>(os, *elem);
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

  template <typename T>
  Tensor<T> read_tensor(std::istream& is)
  {
    TensorFormat format;

    format.baseFormat = read_bytes<std::int32_t>(is);
    format.subFormat = read_bytes<std::int32_t>(is);

    auto expectedFormat = tensorFormat<T>();
    if (format != expectedFormat)
    {
      throw std::runtime_error("Expected tensor format " + format.toString() + " where " + expectedFormat.toString() + " was expected.");
    }

    auto shapeSz = read_bytes<std::uint64_t>(is);
    std::vector<size_t> shape(shapeSz);
    std::vector<size_t> strides(shapeSz);
    for (auto i = 0_uz; i < shapeSz; ++i)
    {
      shape[i] = read_bytes<std::uint64_t>(is);
      strides[i] = read_bytes<std::uint64_t>(is);
    }

    Tensor<T> ret(shape);
    if (ret.strides() != strides)
    {
      throw std::runtime_error("Incorrect strides in saved data (expected " + index_to_string(ret.strides()) + " but got " + index_to_string(strides) + ")");
    }

    auto sz = ret.size();
    for (auto * elem = ret.data(); elem != ret.data() + sz; ++elem)
    {
      *elem = read_element<T>(is);
    }

    return ret;
  }
}

#endif // MASSPCF_TENSOR_IO_H
