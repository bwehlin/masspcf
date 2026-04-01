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

#include <gtest/gtest.h>

#include <mpcf/tensor.hpp>
#include <mpcf/walk.hpp>
#include <mpcf/io/tensor_io.hpp>
#include <mpcf/functional/pcf.hpp>
#include <mpcf/persistence/barcode.hpp>

#include <sstream>
#include <cstring>

namespace
{
  using mpcf::io::detail::TensorFormat;
  using mpcf::io::detail::tensorFormat;
  using mpcf::io::detail::getTensorFormat;

  // ============================================================================
  // TensorFormat mapping for supported core types
  // ============================================================================

  TEST(TensorIoCore, TensorFormatScalarAndCompositeTypes)
  {
    using mpcf::float32_t;
    using mpcf::float64_t;

    {
      auto fmt = tensorFormat<float32_t>();
      EXPECT_EQ(1, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<float64_t>();
      EXPECT_EQ(1, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<mpcf::Pcf<float32_t, float32_t>>();
      EXPECT_EQ(100, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<mpcf::Pcf<float64_t, float64_t>>();
      EXPECT_EQ(100, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<mpcf::PointCloud<float32_t>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<mpcf::PointCloud<float64_t>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<mpcf::ph::Barcode<float32_t>>();
      EXPECT_EQ(10000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<mpcf::ph::Barcode<float64_t>>();
      EXPECT_EQ(10000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
  }

  TEST(TensorIoCore, TensorFormatThrowsOnUnsupportedType)
  {
    struct Unsupported {};
    EXPECT_THROW((void)tensorFormat<Unsupported>(), std::runtime_error);
  }

  // ============================================================================
  // write_contiguous_tensor / read_tensor roundtrip for scalar tensor
  // ============================================================================

  TEST(TensorIoCore, ContiguousScalarTensorRoundtrip)
  {
    using T = mpcf::float32_t;
    mpcf::Tensor<T> t({ 2, 3 });
    T v = static_cast<T>(0);
    t.apply([&v](T& x) { x = v++; });

    std::stringstream ss;
    mpcf::io::detail::write_contiguous_tensor(ss, t);

    std::string all = ss.str();

    std::istringstream iss(all);

    mpcf::io::detail::read_tensor_format(iss);

    auto roundtrip = mpcf::io::detail::read_tensor<T>(iss);
    EXPECT_EQ(roundtrip, t);
  }

  // ============================================================================
  // write_tensor handles non-contiguous views by copying internally
  // ============================================================================

  TEST(TensorIoCore, NonContiguousTensorRoundtripViaWriteTensor)
  {
    using T = double;

    mpcf::Tensor<T> base({ 4, 4 });
    T v = 0;
    base.apply([&v](T& x) { x = v++; });

    // Take a strided view so that is_contiguous() is false
    auto view = base[std::vector<mpcf::Slice>{ mpcf::range(0, 4, 2), mpcf::all() }];
    ASSERT_FALSE(view.is_contiguous());

    std::stringstream ss;
    mpcf::io::detail::write_tensor(ss, view);

    std::string all = ss.str();

    std::istringstream iss(all);

    mpcf::io::detail::read_tensor_format(iss);
    auto rt = mpcf::io::detail::read_tensor<T>(iss);

    EXPECT_EQ(rt.shape(), view.shape());

    mpcf::walk(view, [&](const std::vector<size_t>& idx)
    {
      EXPECT_EQ(rt(idx), view(idx));
    });
  }

  // ============================================================================
  // read_tensor detects stride mismatches
  // ============================================================================

  TEST(TensorIoCore, ReadTensorThrowsOnStrideMismatch)
  {
    using T = double;

    mpcf::Tensor<T> t({ 2, 3 });
    T v = 0;
    t.apply([&v](T& x) { x = v++; });

    std::stringstream ss;
    mpcf::io::detail::write_contiguous_tensor(ss, t);

    std::string all = ss.str();
    ASSERT_GE(all.size(), 8u);

    // Work on the payload that starts after the TensorFormat header
    std::string payload(all.begin() + 8, all.end());

    // Layout in payload:
    // [shapeSz:uint64][shape0:uint64][stride0:uint64][shape1:uint64][stride1:uint64]...
    std::uint64_t* raw = reinterpret_cast<std::uint64_t*>(payload.data());
    std::uint64_t shapeSz = raw[0];
    ASSERT_EQ(shapeSz, 2u);

    // Corrupt first stride (located after shapeSz + shape0 + shape1)
    std::size_t strideIdx = 1 + static_cast<std::size_t>(shapeSz) + 0;
    raw[strideIdx] = 999u;

    std::istringstream iss(payload);

    EXPECT_THROW((void)mpcf::io::detail::read_tensor<T>(iss), std::runtime_error);
  }

} // namespace

