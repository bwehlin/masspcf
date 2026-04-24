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

#include <mpcf/persistence/compute_persistence.hpp>

#include <cstdint>

// simplex_cost_units estimates per-item GPU cost for the hybrid ripser
// dispatcher. The formula caps at binomial(n, min(max_dim + 1, n/2))
// to mirror upstream Ripser++'s memory planner. An earlier version
// used n/2 - 1, which collapsed to 0 for n in {2, 3} (force-routing
// them to CPU) and under-estimated small-n items generally. These
// tests pin the cap at the upstream value.
namespace {

using mpcf::ph::detail::simplex_cost_units;

TEST(SimplexCostUnits, NonPositiveReturnsZero)
{
  EXPECT_EQ(simplex_cost_units(0, 1), 0);
  EXPECT_EQ(simplex_cost_units(1, 1), 0);
  EXPECT_EQ(simplex_cost_units(5, -1), 0);
}

TEST(SimplexCostUnits, SmallNDoesNotCollapseToZero)
{
  // Upstream allocates binomial(n, n/2) for these, so the estimate
  // must be positive. Pre-fix these returned 0.
  EXPECT_EQ(simplex_cost_units(2, 1), 2);   // C(2, 1) = 2
  EXPECT_EQ(simplex_cost_units(3, 1), 3);   // C(3, 1) = 3
  EXPECT_EQ(simplex_cost_units(3, 5), 3);   // cap at n/2=1 -> C(3, 1)
}

TEST(SimplexCostUnits, CapAtNHalfMatchesUpstream)
{
  // k_max = min(max_dim + 1, n / 2). For max_dim >= n/2 - 1, the cap
  // binds and we get binomial(n, n/2).
  EXPECT_EQ(simplex_cost_units(4, 3), 6);    // C(4, 2) = 6
  EXPECT_EQ(simplex_cost_units(5, 1), 10);   // C(5, 2) = 10
  EXPECT_EQ(simplex_cost_units(10, 5), 252); // C(10, 5) = 252
}

TEST(SimplexCostUnits, CapAtMaxDimPlusOneWhenSmaller)
{
  // k_max = max_dim + 1 when max_dim + 1 < n/2.
  EXPECT_EQ(simplex_cost_units(100, 0), 100);      // C(100, 1)
  EXPECT_EQ(simplex_cost_units(100, 1), 4950);     // C(100, 2)
  EXPECT_EQ(simplex_cost_units(20, 2), 1140);      // C(20, 3)
}

TEST(SimplexCostUnits, SaturatesAtInt64MaxForHugeBinomials)
{
  // binomial(200, 100) is astronomical; must saturate, not overflow.
  const auto c = simplex_cost_units(200, 99);
  EXPECT_EQ(c, std::numeric_limits<std::int64_t>::max());
}

}  // namespace
