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

#ifndef MPCF_SAMPLING_KDTREE_H
#define MPCF_SAMPLING_KDTREE_H

#include "../tensor.hpp"

#include <nanoflann.hpp>

#include <cstddef>

namespace mpcf::sampling
{

  /// Adaptor exposing a Tensor<T> of shape (N, D) to nanoflann.
  template <typename T>
  struct PointCloudAdaptor
  {
    const Tensor<T>& cloud;
    size_t nPoints;
    size_t dim;

    explicit PointCloudAdaptor(const Tensor<T>& cloud)
      : cloud(cloud)
      , nPoints(cloud.shape()[0])
      , dim(cloud.shape()[1])
    {
    }

    inline size_t kdtree_get_point_count() const { return nPoints; }

    inline T kdtree_get_pt(size_t idx, size_t d) const
    {
      return cloud({idx, d});
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
  };

  template <typename T>
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<T, PointCloudAdaptor<T>>,
      PointCloudAdaptor<T>,
      -1,    // dimensionality at runtime
      size_t // index type
  >;

}

#endif
