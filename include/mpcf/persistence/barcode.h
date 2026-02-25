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

#ifndef MASSPCF_BARCODE_H
#define MASSPCF_BARCODE_H

#include "persistence_pair.h"

namespace mpcf::ph
{
  template <typename T>
  class Barcode
  {
  public:
    /**
     * Does a 1-1 comparison between two `Barcode` objects. For efficiency reasons, equality is only `true` if the bars
     * occur *in the same order*, even though, mathematically, this doesn't matter.
     * @param rhs The `Barcode` to compare against.
     * @return `true` if all bars are the same and occur in the same order, otherwise `false`.
     */
    [[nodiscard]] bool operator==(const Barcode& rhs) const
    {
      return m_persistencePairs == rhs.m_persistencePairs;
    }

  private:
    std::vector<mpcf::ph::PersistencePair<T>> m_persistencePairs;

  };
}

#endif //MASSPCF_BARCODE_H