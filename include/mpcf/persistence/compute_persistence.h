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

#ifndef MASSPCF_COMPUTE_PERSISTENCE_H
#define MASSPCF_COMPUTE_PERSISTENCE_H

#include "../tensor.h"
#include "barcode.h"
#include "persistence_pair.h"

#include "ripser/ripser.h"

namespace mpcf::ph
{
  template <typename T>
  requires std::is_floating_point_v<T>
  Tensor<Barcode<T>> compute_persistence_euclidean(const Tensor<T>& points)
  {
    Tensor<Barcode<T>> ret;

    //rp::ripser<rp::euclidean_distance_matrix> ripser;

    return ret;
  }
}

#endif //MASSPCF_COMPUTE_PERSISTENCE_H