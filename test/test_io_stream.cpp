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

TEST(IoStream, TestPointRoundtrip)
{
  {
    mpcf::Point_f32 pt(0.5f, 2.5f);

    std::stringstream ss;
    mpcf::io::detail::write_element(ss, pt);

    std::istringstream iss(ss.str());
    auto retPt = mpcf::io::detail::read_element<decltype(pt)>(iss);

    EXPECT_EQ(pt, retPt);
  }

  {
    mpcf::Point_f64 pt(0.5, 2.5);

    std::stringstream ss;
    mpcf::io::detail::write_element(ss, pt);

    std::istringstream iss(ss.str());
    auto retPt = mpcf::io::detail::read_element<decltype(pt)>(iss);

    EXPECT_EQ(pt, retPt);
  }
}
