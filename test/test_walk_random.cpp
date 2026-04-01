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
#include <mpcf/tensor.hpp>
#include <mpcf/walk.hpp>
#include <mpcf/executor.hpp>

#include <random>

TEST(WalkRandom, SameSeedProducesSameValues)
{
  mpcf::Tensor<double> a({3, 4});
  mpcf::Tensor<double> b({3, 4});

  mpcf::DefaultRandomGenerator gen1(42);
  mpcf::DefaultRandomGenerator gen2(42);

  mpcf::walk(a, gen1, [&a](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    a(idx) = dist(engine);
  });

  mpcf::walk(b, gen2, [&b](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    b(idx) = dist(engine);
  });

  EXPECT_TRUE(mpcf::allclose(a, b));
}

TEST(WalkRandom, DifferentSeedsProduceDifferentValues)
{
  mpcf::Tensor<double> a({3, 4});
  mpcf::Tensor<double> b({3, 4});

  mpcf::DefaultRandomGenerator gen1(42);
  mpcf::DefaultRandomGenerator gen2(99);

  mpcf::walk(a, gen1, [&a](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    a(idx) = dist(engine);
  });

  mpcf::walk(b, gen2, [&b](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    b(idx) = dist(engine);
  });

  EXPECT_FALSE(mpcf::allclose(a, b));
}

TEST(WalkRandom, ParallelWalkMatchesSequentialWalk)
{
  mpcf::Tensor<double> seq({80, 10});
  mpcf::Tensor<double> par({80, 10});

  mpcf::DefaultRandomGenerator gen1(123);
  mpcf::DefaultRandomGenerator gen2(123);

  mpcf::walk(seq, gen1, [&seq](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    seq(idx) = dist(engine);
  });

  mpcf::parallel_walk(par, gen2, [&par](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    par(idx) = dist(engine);
  }, mpcf::default_executor());

  EXPECT_TRUE(mpcf::allclose(seq, par));
}

TEST(WalkRandom, ParallelWalkIsDeterministicAcrossRuns)
{
  mpcf::Tensor<double> a({80, 10});
  mpcf::Tensor<double> b({80, 10});

  mpcf::DefaultRandomGenerator gen1(77);
  mpcf::DefaultRandomGenerator gen2(77);

  mpcf::parallel_walk(a, gen1, [&a](const std::vector<size_t>& idx, auto& engine) {
    std::normal_distribution<double> dist(0.0, 1.0);
    a(idx) = dist(engine);
  }, mpcf::default_executor());

  mpcf::parallel_walk(b, gen2, [&b](const std::vector<size_t>& idx, auto& engine) {
    std::normal_distribution<double> dist(0.0, 1.0);
    b(idx) = dist(engine);
  }, mpcf::default_executor());

  EXPECT_TRUE(mpcf::allclose(a, b));
}
