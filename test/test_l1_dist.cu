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

#include <mpcf/functional/pcf.hpp>
#include <mpcf/distance_matrix.hpp>
#include <mpcf/task.hpp>
#include <mpcf/functional/operations.cuh>
#include <mpcf/algorithms/functional/matrix_integrate.hpp>

#ifdef BUILD_WITH_CUDA
#include <mpcf/cuda/cuda_matrix_integrate_api.hpp>
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
  TEST(PcfL1Direct, TwoPointPcfIntegrate_f32)
  {
    mpcf::Pcf_f32 f(std::vector<mpcf::Point_f32>({ {0.f, 3.f}, {1.0f, 0.0f} }));
    mpcf::Pcf_f32 g(std::vector<mpcf::Point_f32>({ {0.f, 1.f}, {2.0f, 0.0f} }));

    auto op = mpcf::OperationL1Dist<float, float>{};
    float result = mpcf::integrate(f, g, [&op](const mpcf::Rectangle<float, float>& rect) {
      return op(rect.top, rect.bottom);
    }, 0.f, std::numeric_limits<float>::max());
    result = op(result);

    EXPECT_FLOAT_EQ(result, 3.f);
  }

  TEST(PcfL1Direct, TwoPointPcfIntegrate_f64)
  {
    mpcf::Pcf_f64 f(std::vector<mpcf::Point_f64>({ {0.0, 3.0}, {1.0, 0.0} }));
    mpcf::Pcf_f64 g(std::vector<mpcf::Point_f64>({ {0.0, 1.0}, {2.0, 0.0} }));

    auto op = mpcf::OperationL1Dist<double, double>{};
    double result = mpcf::integrate(f, g, [&op](const mpcf::Rectangle<double, double>& rect) {
      return op(rect.top, rect.bottom);
    }, 0.0, std::numeric_limits<double>::max());
    result = op(result);

    EXPECT_DOUBLE_EQ(result, 3.0);
  }

  // Parameterized on precision and hardware
  struct TestConfig
  {
    mpcf::Hardware hw;
    bool useF64;

    std::string name() const
    {
      std::string s = (hw == mpcf::Hardware::CUDA) ? "CUDA" : "CPU";
      s += useF64 ? "_f64" : "_f32";
      return s;
    }
  };

  class PcfL1IntegratorFixture : public ::testing::TestWithParam<TestConfig>
  {
  public:
    void compute_l1()
    {
      auto cfg = GetParam();
      std::unique_ptr<mpcf::StoppableTask<void>> task;

      if (cfg.hw == mpcf::Hardware::CUDA)
      {
#ifdef BUILD_WITH_CUDA
        if (mpcf::get_num_cuda_devices() == 0)
        {
          GTEST_SKIP() << "No CUDA devices available";
          return;
        }
        if (cfg.useF64)
          task = mpcf::create_cuda_block_integrate_l1_task(m_dm64, m_pcfs64);
        else
          task = mpcf::create_cuda_block_integrate_l1_task(m_dm32, m_pcfs32);
#else
        GTEST_SKIP() << "CUDA not available";
#endif
      }
      else
      {
        if (cfg.useF64)
        {
          auto op = mpcf::OperationL1Dist<double, double>{};
          task = std::make_unique<mpcf::CpuPairwiseIntegrationTask<decltype(op), decltype(m_pcfs64.cbegin()), mpcf::DistanceMatrix<double>, false>>(
              m_dm64, m_pcfs64.cbegin(), m_pcfs64.cend(), op);
        }
        else
        {
          auto op = mpcf::OperationL1Dist<float, float>{};
          task = std::make_unique<mpcf::CpuPairwiseIntegrationTask<decltype(op), decltype(m_pcfs32.cbegin()), mpcf::DistanceMatrix<float>, false>>(
              m_dm32, m_pcfs32.cbegin(), m_pcfs32.cend(), op);
        }
      }

      task->start_async(mpcf::default_executor());
      task->future().get();
    }

    // f32 data
    std::vector<mpcf::Pcf_f32> m_pcfs32;
    mpcf::DistanceMatrix<float> m_dm32{0};

    // f64 data
    std::vector<mpcf::Pcf_f64> m_pcfs64;
    mpcf::DistanceMatrix<double> m_dm64{0};
  };

  TEST_P(PcfL1IntegratorFixture, EmptyPcfPairL1dist)
  {
    auto cfg = GetParam();
    if (cfg.useF64)
    {
      m_pcfs64.resize(2);
      m_dm64 = mpcf::DistanceMatrix<double>(2);
    }
    else
    {
      m_pcfs32.resize(2);
      m_dm32 = mpcf::DistanceMatrix<float>(2);
    }

    compute_l1();

    if (cfg.useF64)
    {
      EXPECT_DOUBLE_EQ(m_dm64(0, 1), 0.0);
    }
    else
    {
      EXPECT_FLOAT_EQ(m_dm32(0, 1), 0.f);
    }
  }

  TEST_P(PcfL1IntegratorFixture, TwoPointPcfL1dist)
  {
    auto cfg = GetParam();
    if (cfg.useF64)
    {
      m_pcfs64.emplace_back(std::vector<mpcf::Point_f64>({{0.0, 3.0}, {1.0, 0.0}}));
      m_pcfs64.emplace_back(std::vector<mpcf::Point_f64>({{0.0, 1.0}, {2.0, 0.0}}));
      m_dm64 = mpcf::DistanceMatrix<double>(2);
    }
    else
    {
      m_pcfs32.emplace_back(std::vector<mpcf::Point_f32>({{0.f, 3.f}, {1.f, 0.f}}));
      m_pcfs32.emplace_back(std::vector<mpcf::Point_f32>({{0.f, 1.f}, {2.f, 0.f}}));
      m_dm32 = mpcf::DistanceMatrix<float>(2);
    }

    compute_l1();

    if (cfg.useF64)
    {
      EXPECT_DOUBLE_EQ(m_dm64(0, 0), 0.0);
      EXPECT_DOUBLE_EQ(m_dm64(0, 1), 3.0);
      EXPECT_DOUBLE_EQ(m_dm64(1, 0), 3.0);
      EXPECT_DOUBLE_EQ(m_dm64(1, 1), 0.0);
    }
    else
    {
      EXPECT_FLOAT_EQ(m_dm32(0, 0), 0.f);
      EXPECT_FLOAT_EQ(m_dm32(0, 1), 3.f);
      EXPECT_FLOAT_EQ(m_dm32(1, 0), 3.f);
      EXPECT_FLOAT_EQ(m_dm32(1, 1), 0.f);
    }
  }

  INSTANTIATE_TEST_SUITE_P(
      PcfL1Integrator,
      PcfL1IntegratorFixture,
      ::testing::Values(
          TestConfig{mpcf::Hardware::CPU, false},
          TestConfig{mpcf::Hardware::CPU, true},
          TestConfig{mpcf::Hardware::CUDA, false},
          TestConfig{mpcf::Hardware::CUDA, true}
      ),
      [](const testing::TestParamInfo<TestConfig>& info) {
        return info.param.name();
      }
  );
}
