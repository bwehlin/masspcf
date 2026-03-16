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

#include <gtest/gtest.h>

#include <mpcf/functional/pcf.h>
#include <mpcf/task.h>
#include <mpcf/functional/operations.cuh>
#include <mpcf/algorithms/functional/matrix_integrate.h>

#ifdef BUILD_WITH_CUDA
#include <mpcf/cuda/cuda_matrix_integrate_api.h>
#endif

#include <vector>
#include <memory>

#ifdef BUILD_WITH_CUDA
#pragma message("Building tester with CUDA")
#else
#pragma message("Building tester without CUDA")
#endif

namespace
{
  // Direct (non-task) integration test to verify the core algorithm
  TEST(PcfL1Direct, TwoPointPcfIntegrate)
  {
    mpcf::Pcf_f32 f(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 3.f), mpcf::Point_f32(1.0f, 0.0f) }));
    mpcf::Pcf_f32 g(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 1.f), mpcf::Point_f32(2.0f, 0.0f) }));

    auto op = mpcf::OperationL1Dist<float, float>{};
    float result = mpcf::integrate(f, g, [&op](const mpcf::Rectangle<float, float>& rect) {
      return op(rect.top, rect.bottom);
    }, 0.f, std::numeric_limits<float>::max());
    result = op(result);

    EXPECT_EQ(result, 3.f);
  }

  class PcfL1IntegratorFixture : public ::testing::TestWithParam<mpcf::Hardware>
  {
  public:
    std::vector<mpcf::Pcf_f32> pcfs;
    std::vector<float> matrix;
    void allocate_matrix()
    {
      matrix.resize(pcfs.size() * pcfs.size(), -1.f);
    }

    float ij(size_t i, size_t j) const
    {
      return matrix.at(i * pcfs.size() + j);
    }

    void compute_l1()
    {
      auto hw = GetParam();
      std::unique_ptr<mpcf::StoppableTask<void>> task;

      if (hw == mpcf::Hardware::CUDA)
      {
#ifdef BUILD_WITH_CUDA
        if (mpcf::get_num_cuda_devices() == 0)
        {
          GTEST_SKIP() << "No CUDA devices available";
          return;
        }
        task = mpcf::create_cuda_matrix_integrate_l1_task(
            matrix.data(), pcfs, 0.f, std::numeric_limits<float>::max());
#else
        GTEST_SKIP() << "CUDA not available";
#endif
      }
      else
      {
        auto op = mpcf::OperationL1Dist<float, float>{};
        task = std::make_unique<mpcf::MatrixIntegrateCpuTask<decltype(op), decltype(pcfs.cbegin())>>(
            matrix.data(), pcfs.cbegin(), pcfs.cend(), op);
      }

      task->start_async(mpcf::default_executor());
      task->future().get();
    }
  };
}

TEST_P(PcfL1IntegratorFixture, EmptyPcfPairL1dist)
{
  pcfs.resize(2);
  allocate_matrix();

  compute_l1();

  EXPECT_EQ(ij(0, 0), 0.f);
  EXPECT_EQ(ij(0, 1), 0.f);
  EXPECT_EQ(ij(1, 0), 0.f);
  EXPECT_EQ(ij(1, 1), 0.f);
}

TEST_P(PcfL1IntegratorFixture, TwoPointPcfL1dist)
{
  pcfs.emplace_back(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 3.f), mpcf::Point_f32(1.0, 0.0) }));
  pcfs.emplace_back(std::vector<mpcf::Point_f32>({ mpcf::Point_f32(0.f, 1.f), mpcf::Point_f32(2.0, 0.0) }));
  allocate_matrix();

  compute_l1();

  EXPECT_EQ(ij(0, 0), 0.f);
  EXPECT_EQ(ij(0, 1), 3.f);
  EXPECT_EQ(ij(1, 0), 3.f);
  EXPECT_EQ(ij(1, 1), 0.f);
}

INSTANTIATE_TEST_CASE_P(
    PcfL1Integrator,
    PcfL1IntegratorFixture,
    ::testing::Values(mpcf::Hardware::CPU, mpcf::Hardware::CUDA),
    [](const testing::TestParamInfo<mpcf::Hardware>& info) {
      switch (info.param)
      {
      case mpcf::Hardware::CPU: return "CPU";
      case mpcf::Hardware::CUDA: return "CUDA";
      default: return "<<UNKNOWN>>";
      }
    }
);
