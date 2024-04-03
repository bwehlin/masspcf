/*
* Copyright 2024 Bjorn Wehlin
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

#include <gtest/gtest.h>

#include <mpcf/pcf.h>
#include <mpcf/algorithms/reduce.h>

namespace
{
  void PrintTo(const mpcf::Pcf_f64& f, std::ostream* os)
  {
    *os << f.to_string();
  } 
}

TEST(ParallelReduce, AddThreeFunctions)
{
  std::vector<mpcf::Pcf_f64> pcfs;

  pcfs.emplace_back(mpcf::Pcf_f64{ {0., 3.}, {1., 2.}, {4., 5.}, {6., 0.} });
  pcfs.emplace_back(mpcf::Pcf_f64{ {0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}, {8., 3.} });
  pcfs.emplace_back(mpcf::Pcf_f64{ {0., 0.}, {3., 7.}, {5., 2.} });

  auto res = mpcf::parallel_reduce(pcfs.begin(), pcfs.end(), [](const typename mpcf::Pcf_f64::rectangle_type& rect) {
    return rect.top + rect.bottom;
    });

  EXPECT_EQ(res, (pcfs[0] + pcfs[1] + pcfs[2]));
}
