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

#ifndef MPCF_ALGORITHMS_MATRIX_INTEGRATE_H
#define MPCF_ALGORITHMS_MATRIX_INTEGRATE_H

#include "iterate_rectangles.hpp"
#include "../pcf_chunk_precompute.hpp"
#include "../../functional/pcf.hpp"
#include "../../distance_matrix.hpp"
#include "../../symmetric_matrix.hpp"
#include "../../tensor.hpp"
#include "../../executor.hpp"
#include "../../settings.hpp"
#include "../../task.hpp"

#include <optional>
#include <vector>

#include <taskflow/algorithm/for_each.hpp>

namespace mpcf
{
  template <typename Tt, typename Tv, typename PairOp>
  Tv integrate(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g, PairOp op, Tt a = Tt(0), Tt b = std::numeric_limits<Tt>::max())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;

    Tv val = Tv(0);
    iterate_rectangles(f.points(), g.points(), [&val, &op](const rect_t& rect) -> void {
      val += (rect.right - rect.left) * op(rect.f_value, rect.g_value);
    }, a, b);

    return val;
  }

  /// Integrate two PCFs with tail acceleration.
  ///
  /// Runs the same merge-scan as integrate(), but when one PCF exhausts
  /// at the common final value, switches to summing precomputed chunk
  /// integrals for the remainder instead of stepping breakpoint by
  /// breakpoint.
  template <typename Tt, typename Tv, typename PairOp>
  Tv integrate_with_tail_accel(
      const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g,
      PairOp op, Tt a, Tt b,
      const PcfChunkData<Tv>& fChunks, size_t fIdx,
      const PcfChunkData<Tv>& gChunks, size_t gIdx)
  {
    auto const& fpts = f.points();
    auto const& gpts = g.points();

    auto fi = max_time_iterator_prior_to(fpts.begin(), fpts.end(), a);
    auto gi = max_time_iterator_prior_to(gpts.begin(), gpts.end(), a);

    Tt t = a;
    Tv ret = Tv(0);
    Tv cv = fChunks.commonFinalValue;
    size_t chunkSize = fChunks.chunkSize;

    while (t < b)
    {
      Tt tprev = t;
      Tv fv = fi->v;
      Tv gv = gi->v;

      auto fi_next = std::next(fi);
      auto gi_next = std::next(gi);

      if (fi_next != fpts.end() && gi_next != gpts.end())
      {
        auto delta = fi_next->t - gi_next->t;
        if (delta <= 0) fi = fi_next;
        if (delta >= 0) gi = gi_next;
      }
      else if (fi_next != fpts.end())
      {
        fi = fi_next;
      }
      else if (gi_next != gpts.end())
      {
        gi = gi_next;
      }
      else
      {
        ret += (b - tprev) * op(fv, gv);
        break;
      }

      t = std::min(std::max(fi->t, gi->t), b);
      ret += (t - tprev) * op(fv, gv);

      // --- Tail acceleration ---
      auto fi_next2 = std::next(fi);
      auto gi_next2 = std::next(gi);

      if (fi_next2 == fpts.end() && gi_next2 != gpts.end())
      {
        // f exhausted — skip remaining g via chunks
        Tt fLastTime = fi->t;
        size_t giIdx = static_cast<size_t>(std::distance(gpts.begin(), gi));

        // Step to chunk boundary past clipping point
        while (std::next(gi) != gpts.end())
        {
          if (giIdx % chunkSize == 0 && gi->t >= fLastTime)
            break;
          tprev = t;
          gv = gi->v;
          gi = std::next(gi);
          ++giIdx;
          t = std::min(std::max(fLastTime, gi->t), b);
          ret += (t - tprev) * op(cv, gv);
          if (t >= b) return op(ret);
        }

        // Sum remaining precomputed chunks
        if (std::next(gi) != gpts.end())
        {
          size_t firstChunk = giIdx / chunkSize;
          size_t gChunkOff = gChunks.offsets[gIdx];
          size_t nChunks = gChunks.offsets[gIdx + 1] - gChunkOff;
          for (size_t c = firstChunk; c < nChunks; ++c)
            ret += gChunks.values[gChunkOff + c];
        }

        ret += (b - gpts.back().t) * op(cv, gpts.back().v);
        break;
      }

      if (gi_next2 == gpts.end() && fi_next2 != fpts.end())
      {
        // g exhausted — skip remaining f via chunks
        Tt gLastTime = gi->t;
        size_t fiIdx = static_cast<size_t>(std::distance(fpts.begin(), fi));

        while (std::next(fi) != fpts.end())
        {
          if (fiIdx % chunkSize == 0 && fi->t >= gLastTime)
            break;
          tprev = t;
          fv = fi->v;
          fi = std::next(fi);
          ++fiIdx;
          t = std::min(std::max(gLastTime, fi->t), b);
          ret += (t - tprev) * op(fv, cv);
          if (t >= b) return op(ret);
        }

        if (std::next(fi) != fpts.end())
        {
          size_t firstChunk = fiIdx / chunkSize;
          size_t fChunkOff = fChunks.offsets[fIdx];
          size_t nChunks = fChunks.offsets[fIdx + 1] - fChunkOff;
          for (size_t c = firstChunk; c < nChunks; ++c)
            ret += fChunks.values[fChunkOff + c];
        }

        ret += (b - fpts.back().t) * op(fpts.back().v, cv);
        break;
      }
    }

    return ret;
  }

  /// CPU pairwise integration task (pdist / l2_kernel).
  /// OutputT must support operator()(i, j) = val (DistanceMatrix or SymmetricMatrix).
  /// When includeDiagonal is false, computes i > j only (DistanceMatrix).
  /// When includeDiagonal is true, computes i >= j (SymmetricMatrix).
  template <typename TOperation, typename PcfFwdIt, typename OutputT, bool includeDiagonal>
  class CpuPairwiseIntegrationTask : public mpcf::StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfFwdIt::value_type;
    using value_type = typename pcf_type::value_type;
    using time_type = typename pcf_type::time_type;

    CpuPairwiseIntegrationTask(OutputT out, PcfFwdIt beginPcfs, PcfFwdIt endPcfs, TOperation op)
      : m_fs(beginPcfs, endPcfs)
      , m_out(std::move(out))
      , m_op(op)
    { }

    OutputT& output() { return m_out; }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto sz = m_fs.size();
      size_t totalWork;
      if constexpr (includeDiagonal)
        totalWork = (sz * (sz + 1)) / 2;
      else
        totalWork = (sz * (sz - 1)) / 2;

      next_step(totalWork, includeDiagonal ? "Computing kernel matrix." : "Computing distance matrix.", "integral");

      // Tail acceleration
      auto& s = mpcf::settings();
      if (s.tailAccelChunkSize > 0)
      {
        auto cv = find_common_final_value(m_fs);
        if (cv)
        {
          m_chunkData = precompute_chunks(m_fs, m_op, *cv, s.tailAccelChunkSize);
        }
      }

      tf::Taskflow flow;
      if (m_fs.empty())
      {
        return exec.cpu()->run(std::move(flow));
      }

      flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [this](size_t i) {
        if (stop_requested())
        {
          return;
        }
        compute_row(i);
      });

      return exec.cpu()->run(std::move(flow));
    }

    void compute_row(size_t i)
    {
      auto sz = m_fs.size();
      size_t jStart = includeDiagonal ? i : i + 1;
      for (size_t j = jStart; j < sz; ++j)
      {
        value_type val;
        if (m_chunkData)
        {
          val = m_op(integrate_with_tail_accel<time_type, value_type>(
              m_fs[i], m_fs[j], m_op, 0, std::numeric_limits<time_type>::max(),
              *m_chunkData, i, *m_chunkData, j));
        }
        else
        {
          val = m_op(integrate<time_type, value_type>(m_fs[i], m_fs[j], m_op, 0, std::numeric_limits<time_type>::max()));
        }
        m_out(i, j) = val;
      }
      add_progress(sz - jStart);
    }

    std::vector<pcf_type> m_fs;
    OutputT m_out;
    TOperation m_op;
    std::optional<PcfChunkData<value_type>> m_chunkData;
  };

  /// CPU cross-distance integration task (cdist).
  /// Computes all (i,j) pairs between two separate function sets.
  template <typename TOperation, typename PcfFwdIt>
  class CpuCrossIntegrationTask : public mpcf::StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfFwdIt::value_type;
    using value_type = typename pcf_type::value_type;
    using time_type = typename pcf_type::time_type;

    CpuCrossIntegrationTask(Tensor<value_type> out,
                 PcfFwdIt beginRows, PcfFwdIt endRows,
                 PcfFwdIt beginCols, PcfFwdIt endCols,
                 TOperation op)
      : m_rowFs(beginRows, endRows)
      , m_colFs(beginCols, endCols)
      , m_out(std::move(out))
      , m_op(op)
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto nRows = m_rowFs.size();
      auto nCols = m_colFs.size();

      next_step(nRows * nCols, "Computing cross-distances.", "integral");

      // Tail acceleration
      auto& s = mpcf::settings();
      if (s.tailAccelChunkSize > 0)
      {
        auto rcv = find_common_final_value(m_rowFs);
        auto ccv = find_common_final_value(m_colFs);
        if (rcv && ccv && *rcv == *ccv)
        {
          m_rowChunkData = precompute_chunks(m_rowFs, m_op, *rcv, s.tailAccelChunkSize);
          m_colChunkData = precompute_chunks(m_colFs, m_op, *ccv, s.tailAccelChunkSize);
        }
      }

      tf::Taskflow flow;
      if (nRows == 0 || nCols == 0)
      {
        return exec.cpu()->run(std::move(flow));
      }

      value_type* data = m_out.data();
      flow.for_each_index<size_t, size_t, size_t>(0ul, nRows, 1ul, [this, data, nCols](size_t i) {
        if (stop_requested())
        {
          return;
        }
        for (size_t j = 0; j < nCols; ++j)
        {
          value_type val;
          if (m_rowChunkData && m_colChunkData)
          {
            val = m_op(integrate_with_tail_accel<time_type, value_type>(
                m_rowFs[i], m_colFs[j], m_op, 0, std::numeric_limits<time_type>::max(),
                *m_rowChunkData, i, *m_colChunkData, j));
          }
          else
          {
            val = m_op(integrate<time_type, value_type>(
                m_rowFs[i], m_colFs[j], m_op, 0, std::numeric_limits<time_type>::max()));
          }
          data[i * nCols + j] = val;
        }
        add_progress(nCols);
      });

      return exec.cpu()->run(std::move(flow));
    }

    std::vector<pcf_type> m_rowFs;
    std::vector<pcf_type> m_colFs;
    Tensor<value_type> m_out;
    TOperation m_op;
    std::optional<PcfChunkData<value_type>> m_rowChunkData;
    std::optional<PcfChunkData<value_type>> m_colChunkData;
  };

}

#endif
