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

// Tail-acceleration chunk precomputation for piecewise constant functions.
//
// When all PCFs share a common final breakpoint value v_common, we can
// precompute per-chunk integrals of op(v_common, v_j) over groups of
// breakpoint intervals.  When the merge-scan detects that one PCF is
// exhausted at value v_common, it sums precomputed chunks instead of
// stepping through the remaining breakpoints one by one.
//
// Pure C++ — no CUDA dependency.  Works with any element type that has
// .t and .v members (SimplePoint for GPU, TimePoint for CPU).

#ifndef MPCF_ALGORITHMS_PCF_CHUNK_PRECOMPUTE_HPP
#define MPCF_ALGORITHMS_PCF_CHUNK_PRECOMPUTE_HPP

#include <cstddef>
#include <optional>
#include <vector>

namespace mpcf
{

  /// Precomputed per-chunk integrals for tail acceleration.
  /// Only created when all PCFs share a common final value.
  template <typename Tv>
  struct PcfChunkData
  {
    std::vector<size_t> offsets;  ///< Per-PCF offset into values (size: nPcfs+1)
    std::vector<Tv> values;       ///< Flat array of chunk integrals
    size_t chunkSize;              ///< Number of breakpoint intervals per chunk
    Tv commonFinalValue;           ///< The shared final breakpoint value
  };

  // ---------------------------------------------------------------
  // Overloads for flattened offset + element arrays (GPU path)
  // ---------------------------------------------------------------

  /// Check whether every PCF in a flattened offset/element array shares the
  /// same last-breakpoint value.  Returns the common value or nullopt.
  /// PCFs with zero breakpoints are skipped (they have no final value and
  /// contribute no intervals).
  template <typename Tv, typename ElementT>
  std::optional<Tv> find_common_final_value(
      const std::vector<size_t>& offsets,
      const std::vector<ElementT>& elements,
      size_t nPcfs)
  {
    if (nPcfs == 0) return std::nullopt;

    std::optional<Tv> common;

    for (size_t i = 0; i < nPcfs; ++i)
    {
      size_t start = offsets[i];
      size_t end   = offsets[i + 1];
      if (end == start) continue;  // empty — skip

      Tv lastVal = static_cast<Tv>(elements[end - 1].v);
      if (!common)
      {
        common = lastVal;
      }
      else if (lastVal != *common)
      {
        return std::nullopt;
      }
    }

    return common;
  }

  /// Precompute chunk integrals from flattened offset/element arrays.
  ///
  /// Each chunk k of PCF i stores:
  ///   Σ_{j in chunk k} (t_{j+1} - t_j) * op(v_common, v_j)
  ///
  /// The interval from the last breakpoint to b is NOT included because
  /// b is a runtime parameter; the kernel handles it separately.
  template <typename Tv, typename ElementT, typename ComboOp>
  PcfChunkData<Tv> precompute_chunks(
      const std::vector<size_t>& offsets,
      const std::vector<ElementT>& elements,
      size_t nPcfs,
      ComboOp op, Tv commonFinalValue, size_t chunkSize)
  {
    PcfChunkData<Tv> result;
    result.chunkSize = chunkSize;
    result.commonFinalValue = commonFinalValue;
    result.offsets.resize(nPcfs + 1);

    size_t totalChunks = 0;
    for (size_t i = 0; i < nPcfs; ++i)
    {
      size_t nBkpts = offsets[i + 1] - offsets[i];
      size_t nIntervals = nBkpts > 1 ? nBkpts - 1 : 0;
      result.offsets[i] = totalChunks;
      totalChunks += (nIntervals + chunkSize - 1) / chunkSize;
    }
    result.offsets[nPcfs] = totalChunks;
    result.values.assign(totalChunks, Tv(0));

    for (size_t i = 0; i < nPcfs; ++i)
    {
      size_t bkptStart = offsets[i];
      size_t nBkpts = offsets[i + 1] - bkptStart;
      if (nBkpts <= 1) continue;

      size_t chunkOffset = result.offsets[i];
      for (size_t j = 0; j + 1 < nBkpts; ++j)
      {
        size_t cIdx = j / chunkSize;
        auto dt = elements[bkptStart + j + 1].t - elements[bkptStart + j].t;
        result.values[chunkOffset + cIdx] += dt * op(commonFinalValue, static_cast<Tv>(elements[bkptStart + j].v));
      }
    }

    return result;
  }

  // ---------------------------------------------------------------
  // Overloads for PCF vectors (CPU path)
  // ---------------------------------------------------------------

  /// Check whether every PCF in a vector shares the same last-breakpoint
  /// value.  PCFs with zero points are skipped.
  template <typename PcfT>
  std::optional<typename PcfT::value_type> find_common_final_value(
      const std::vector<PcfT>& pcfs)
  {
    using Tv = typename PcfT::value_type;

    std::optional<Tv> common;
    for (auto const& f : pcfs)
    {
      if (f.size() == 0) continue;
      Tv lastVal = f.points().back().v;
      if (!common)
        common = lastVal;
      else if (lastVal != *common)
        return std::nullopt;
    }
    return common;
  }

  /// Precompute chunk integrals from a PCF vector.
  template <typename PcfT, typename ComboOp>
  PcfChunkData<typename PcfT::value_type> precompute_chunks(
      const std::vector<PcfT>& pcfs,
      ComboOp op,
      typename PcfT::value_type commonFinalValue,
      size_t chunkSize)
  {
    using Tv = typename PcfT::value_type;

    PcfChunkData<Tv> result;
    result.chunkSize = chunkSize;
    result.commonFinalValue = commonFinalValue;

    size_t nPcfs = pcfs.size();
    result.offsets.resize(nPcfs + 1);

    size_t totalChunks = 0;
    for (size_t i = 0; i < nPcfs; ++i)
    {
      size_t nIntervals = pcfs[i].size() > 1 ? pcfs[i].size() - 1 : 0;
      result.offsets[i] = totalChunks;
      totalChunks += (nIntervals + chunkSize - 1) / chunkSize;
    }
    result.offsets[nPcfs] = totalChunks;
    result.values.assign(totalChunks, Tv(0));

    for (size_t i = 0; i < nPcfs; ++i)
    {
      auto const& pts = pcfs[i].points();
      size_t chunkOffset = result.offsets[i];
      for (size_t j = 0; j + 1 < pts.size(); ++j)
      {
        size_t cIdx = j / chunkSize;
        auto dt = pts[j + 1].t - pts[j].t;
        result.values[chunkOffset + cIdx] += dt * op(commonFinalValue, pts[j].v);
      }
    }

    return result;
  }

} // namespace mpcf

#endif
