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

#include <../include/mpcf/io/io_stream.h>
#include <mpcf/io.h>
#include <mpcf/tensor.h>

#include <sstream>
#include <iostream>

namespace mpcf
{
  template <typename T>
  void PrintTo(const mpcf::Tensor<T>& tensor, std::ostream* os)
  {
    *os << "Tensor[\n";
    tensor.walk([&tensor, os](const std::vector<size_t>& index) {

      *os << "  " << index_to_string(index) << ": " << tensor(index) << '\n';

    });
    *os << "]";
  }
}

std::string to_printable(const std::string& in)
{
  std::stringstream out;

  out << "String of length " << in.length() << ": >>";

  for (auto i = 0_uz; i < in.length(); ++i)
  {
    if (std::isprint(in[i]))
    {
      out << in[i];
    }
    else
    {
      out << "\\" << static_cast<unsigned int>(in[i]);
    }
  }
  out << "<<";
  return out.str();
}

#if 0
TEST(IoStream, GoAroundHasCorrectDataTypes)
{
  std::stringstream ss("", std::ios::out | std::ios::binary);

  mpcf::Tensor<mpcf::float64_t> dblTensor;

  mpcf::write(dblTensor, ss);

  //ASSERT_TRUE(false) << to_printable(ss.str());

  //os << dblTensor;

  //std::string data = ss.str();
  //std::istringstream iss(data, std::ios::in | std::ios::binary);

  //mpcf::IStream is(iss);



}
#endif

template <typename T>
class IoStreamTest : public ::testing::Test {};

namespace
{
  using FloatTypes = ::testing::Types<mpcf::float32_t, mpcf::float64_t>;
  TYPED_TEST_SUITE(IoStreamTest, FloatTypes);

  TYPED_TEST(IoStreamTest, TestPointRoundtrip)
  {
    using PointT = mpcf::Point<TypeParam, TypeParam>;
    PointT pt(0.5, 2.5);

    std::stringstream ss;
    mpcf::io::detail::write_element(ss, pt);

    std::istringstream iss(ss.str());
    auto retPt = mpcf::io::detail::read_element<PointT>(iss);

    EXPECT_EQ(pt, retPt);
  }

  TYPED_TEST(IoStreamTest, TestPcfRoundtrip)
  {
    using PcfT = mpcf::Pcf<TypeParam, TypeParam>;

    std::vector<typename PcfT::point_type> pts({ { 0., 10. }, { 1., 20. }, { 2., 30. } });
    PcfT pcf(std::move(pts));

    std::stringstream ss;
    mpcf::io::detail::write_element(ss, pcf);

    std::istringstream iss(ss.str());
    auto retPcf = mpcf::io::detail::read_element<PcfT>(iss);

    EXPECT_EQ(pcf, retPcf);
  }

  TYPED_TEST(IoStreamTest, TestFloatTensorRoundtrip)
  {
    using TensorT = mpcf::Tensor<TypeParam>;

    TensorT tensor({ 2, 3, 4 });

    tensor.walk([&tensor](const std::vector<size_t>& idx) {
      tensor(idx) = 100 * idx[0] + 10 * idx[1] + idx[2];
    });

    std::stringstream ss;
    mpcf::io::detail::write_tensor(ss, tensor);

    std::istringstream iss(ss.str());
    TensorT retTensor = mpcf::io::detail::read_tensor<TypeParam>(iss);

    EXPECT_EQ(tensor, retTensor);
  }
}