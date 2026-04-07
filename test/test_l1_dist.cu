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
  template <typename T>
  class PcfL1DirectTest : public ::testing::Test {};

  using DirectTypes = ::testing::Types<float, double>;
  TYPED_TEST_SUITE(PcfL1DirectTest, DirectTypes);

  TYPED_TEST(PcfL1DirectTest, TwoPointPcfIntegrate)
  {
    using T = TypeParam;
    mpcf::Pcf<T, T> f(std::vector<mpcf::TimePoint<T, T>>({ {T(0), T(3)}, {T(1), T(0)} }));
    mpcf::Pcf<T, T> g(std::vector<mpcf::TimePoint<T, T>>({ {T(0), T(1)}, {T(2), T(0)} }));

    auto op = mpcf::OperationL1Dist<T, T>{};
    T result = op(mpcf::integrate(f, g, op));

    EXPECT_NEAR(result, T(3), T(1e-6));
  }

  // Parameterized on precision and hardware
  template <typename T, mpcf::Hardware Hw>
  struct TestConfig
  {
    using value_type = T;
    static constexpr mpcf::Hardware hw = Hw;
  };

  template <typename Cfg>
  class PcfL1IntegratorFixture : public ::testing::Test
  {
  public:
    using T = typename Cfg::value_type;
    using PcfT = mpcf::Pcf<T, T>;

    void compute_l1()
    {
      std::unique_ptr<mpcf::StoppableTask<void>> task;

      if constexpr (Cfg::hw == mpcf::Hardware::CUDA)
      {
#ifdef BUILD_WITH_CUDA
        if (mpcf::get_num_cuda_devices() == 0)
        {
          GTEST_SKIP() << "No CUDA devices available";
          return;
        }
        task = mpcf::create_cuda_block_integrate_l1_task(m_dm, m_pcfs);
#else
        GTEST_SKIP() << "CUDA not available";
#endif
      }
      else
      {
        auto op = mpcf::OperationL1Dist<T, T>{};
        task = std::make_unique<mpcf::CpuPairwiseIntegrationTask<decltype(op), decltype(m_pcfs.cbegin()), mpcf::DistanceMatrix<T>, false>>(
            m_dm, m_pcfs.cbegin(), m_pcfs.cend(), op);
      }

      task->start_async(mpcf::default_executor());
      task->future().get();
    }

    std::vector<PcfT> m_pcfs;
    mpcf::DistanceMatrix<T> m_dm{0};
  };

  using IntegratorConfigs = ::testing::Types<
      TestConfig<float, mpcf::Hardware::CPU>,
      TestConfig<double, mpcf::Hardware::CPU>,
      TestConfig<float, mpcf::Hardware::CUDA>,
      TestConfig<double, mpcf::Hardware::CUDA>
  >;
  TYPED_TEST_SUITE(PcfL1IntegratorFixture, IntegratorConfigs);

  TYPED_TEST(PcfL1IntegratorFixture, EmptyPcfPairL1dist)
  {
    using T = typename TypeParam::value_type;
    this->m_pcfs.resize(2);
    this->m_dm = mpcf::DistanceMatrix<T>(2);

    this->compute_l1();

    EXPECT_NEAR(this->m_dm(0, 1), T(0), T(1e-6));
  }

  TYPED_TEST(PcfL1IntegratorFixture, TwoPointPcfL1dist)
  {
    using T = typename TypeParam::value_type;
    using PointT = mpcf::TimePoint<T, T>;

    this->m_pcfs.emplace_back(std::vector<PointT>({{T(0), T(3)}, {T(1), T(0)}}));
    this->m_pcfs.emplace_back(std::vector<PointT>({{T(0), T(1)}, {T(2), T(0)}}));
    this->m_dm = mpcf::DistanceMatrix<T>(2);

    this->compute_l1();

    EXPECT_NEAR(this->m_dm(0, 0), T(0), T(1e-6));
    EXPECT_NEAR(this->m_dm(0, 1), T(3), T(1e-6));
    EXPECT_NEAR(this->m_dm(1, 0), T(3), T(1e-6));
    EXPECT_NEAR(this->m_dm(1, 1), T(0), T(1e-6));
  }
}
