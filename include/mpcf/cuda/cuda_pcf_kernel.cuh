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

// CUDA kernel for piecewise constant function integration.
// Contains the rectangle iteration device function, the block
// integration kernel, and its host-side launch wrapper.

#ifndef MPCF_CUDA_PCF_KERNEL_CUH
#define MPCF_CUDA_PCF_KERNEL_CUH

#include <cuda_runtime.h>

#include "cuda_matrix_integrate_structs.cuh"
#include "cuda_util.cuh"
#include "triangle_skip_mode.hpp"

namespace mpcf
{

  namespace internal
  {
    template <typename Tt, typename Tv>
    struct PcfBlockKernelParams
    {
      Tv* __restrict__ matrix;
      const size_t* __restrict__ rowTimePointOffsets;
      const size_t* __restrict__ colTimePointOffsets;
      const SimplePoint<Tt, Tv>* __restrict__ rowPoints;
      const SimplePoint<Tt, Tv>* __restrict__ colPoints;
      size_t nRows;
      size_t nCols;
      size_t globalRowStart;
      size_t globalColStart;
      TriangleSkipMode skipMode;
    };

    /// Walk two PCFs simultaneously through their breakpoints,
    /// calling cb(left, right, fValue, gValue) for each rectangle.
    template<typename Tt, typename Tv, typename RectangleCallback>
    __device__ void cuda_pcf_iterate_rectangles(
        const SimplePoint<Tt, Tv>* rowPoints, size_t fOffset, size_t fsz,
        const SimplePoint<Tt, Tv>* colPoints, size_t gOffset, size_t gsz,
        Tt a, Tt b, RectangleCallback cb)
    {
      Tt t = a;
      Tt tprev = t;

      Tv fv;
      Tv gv;

      size_t fi = 0;
      size_t gi = 0;

      const SimplePoint<Tt, Tv>* fpts = rowPoints + fOffset;
      const SimplePoint<Tt, Tv>* gpts = colPoints + gOffset;

      while (t < b)
      {
        tprev = t;
        fv = fpts[fi].v;
        gv = gpts[gi].v;

        if (fi + 1 < fsz && gi + 1 < gsz)
        {
          auto delta = fpts[fi + 1].t - gpts[gi + 1].t;
          if (delta <= 0)
          {
            ++fi;
          }
          if (delta >= 0)
          {
            ++gi;
          }
        }
        else
        {
          if (fi + 1 < fsz)
          {
            ++fi;
          }
          else if (gi + 1 < gsz)
          {
            ++gi;
          }
          else
          {
            cb(tprev, b, fv, gv);
            return;
          }
        }

        t = max(fpts[fi].t, gpts[gi].t);
        cb(tprev, t, fv, gv);
      }
    }

    /// CUDA kernel: integrate one PCF pair per thread across a 2D block.
    template <typename Tt, typename Tv, typename ComboOp>
    __global__
    void cuda_pcf_block_integrate(
        PcfBlockKernelParams<Tt, Tv> params,
        Tt a, Tt b, ComboOp op)
    {
      size_t iLocal = blockDim.x * blockIdx.x + threadIdx.x;
      size_t jLocal = blockDim.y * blockIdx.y + threadIdx.y;

      if (iLocal >= params.nRows || jLocal >= params.nCols)
      {
        return;
      }

      size_t iGlobal = iLocal + params.globalRowStart;
      size_t jGlobal = jLocal + params.globalColStart;

      if (params.skipMode == TriangleSkipMode::LowerTriangleSkipDiag && iGlobal <= jGlobal)
      {
        return;
      }
      if (params.skipMode == TriangleSkipMode::LowerTriangle && iGlobal < jGlobal)
      {
        return;
      }

      size_t fOffset = params.rowTimePointOffsets[iLocal];
      size_t fsz = params.rowTimePointOffsets[iLocal + 1] - fOffset;

      size_t gOffset = params.colTimePointOffsets[jLocal];
      size_t gsz = params.colTimePointOffsets[jLocal + 1] - gOffset;

      Tv ret = 0;
      cuda_pcf_iterate_rectangles<Tt, Tv>(
          params.rowPoints, fOffset, fsz,
          params.colPoints, gOffset, gsz,
          a, b, [&ret, op](Tt l, Tt r, Tv f, Tv g) {
            ret += (r - l) * op(f, g);
          });

      params.matrix[iLocal * params.nCols + jLocal] = op(ret);
    }

    /// Chunk size for shared-memory tiling of row PCF breakpoints.
    static constexpr size_t PCF_SMEM_CHUNK_SIZE = 256;

    /// CUDA kernel: integrate PCF pairs with shared-memory tiling.
    ///
    /// Row PCF breakpoints are loaded into shared memory in chunks of
    /// ChunkSize entries, so arbitrarily large PCFs are handled without
    /// exceeding the shared memory budget.  Each CUDA block processes
    /// blockDim.x rows x blockDim.y columns; the blockDim.y column
    /// threads sharing a row cooperatively load that row's chunk and
    /// then read breakpoints from fast shared memory during the
    /// merge-scan inner loop.
    ///
    /// Dynamic shared memory layout:
    ///   SimplePoint<Tt,Tv>[blockDim.x * (ChunkSize+1)]  row chunks
    ///   size_t[blockDim.x + 1]                            row offsets
    template <size_t ChunkSize, typename Tt, typename Tv, typename ComboOp>
    __global__
    void cuda_pcf_block_integrate_smem(
        PcfBlockKernelParams<Tt, Tv> params,
        Tt a, Tt b, ComboOp op)
    {
      extern __shared__ char smem_raw[];

      constexpr size_t kStride = ChunkSize + 1;
      auto* smemRowPts = reinterpret_cast<SimplePoint<Tt, Tv>*>(smem_raw);
      auto* smemRowOffsets = reinterpret_cast<size_t*>(
          smem_raw + blockDim.x * kStride * sizeof(SimplePoint<Tt, Tv>));

      size_t iLocal = blockDim.x * blockIdx.x + threadIdx.x;
      size_t jLocal = blockDim.y * blockIdx.y + threadIdx.y;
      size_t flatTid = threadIdx.x * blockDim.y + threadIdx.y;
      size_t nThreads = blockDim.x * blockDim.y;

      // --- Load row offsets into shared memory ---
      size_t baseRow = static_cast<size_t>(blockDim.x) * blockIdx.x;
      size_t remaining = params.nRows > baseRow ? params.nRows - baseRow : 0;
      size_t actualRows = static_cast<size_t>(blockDim.x) < remaining
          ? static_cast<size_t>(blockDim.x) : remaining;

      for (size_t k = flatTid; k <= actualRows; k += nThreads)
      {
        smemRowOffsets[k] = params.rowTimePointOffsets[baseRow + k];
      }
      __syncthreads();

      // --- Per-thread activity and offset setup ---
      bool active = (iLocal < params.nRows && jLocal < params.nCols);

      if (active)
      {
        size_t iGlobal = iLocal + params.globalRowStart;
        size_t jGlobal = jLocal + params.globalColStart;
        if (params.skipMode == TriangleSkipMode::LowerTriangleSkipDiag && iGlobal <= jGlobal)
          active = false;
        if (params.skipMode == TriangleSkipMode::LowerTriangle && iGlobal < jGlobal)
          active = false;
      }

      size_t fOffset = 0;
      size_t fsz = 0;
      if (threadIdx.x < actualRows)
      {
        fOffset = smemRowOffsets[threadIdx.x];
        fsz = smemRowOffsets[threadIdx.x + 1] - fOffset;
      }

      size_t gOffset = 0;
      size_t gsz = 0;
      if (active)
      {
        gOffset = params.colTimePointOffsets[jLocal];
        gsz = params.colTimePointOffsets[jLocal + 1] - gOffset;
      }

      // --- Max row PCF size in this block (determines chunk count) ---
      __shared__ size_t maxFsz;
      if (flatTid == 0)
      {
        size_t mx = 0;
        for (size_t r = 0; r < actualRows; ++r)
        {
          size_t sz = smemRowOffsets[r + 1] - smemRowOffsets[r];
          if (sz > mx) mx = sz;
        }
        maxFsz = mx;
      }
      __syncthreads();

      // --- Merge-scan state (thread-local, persists across chunks) ---
      const SimplePoint<Tt, Tv>* __restrict__ gpts = params.colPoints + gOffset;
      Tt t = a;
      Tv ret = 0;
      size_t fi = 0;
      size_t gi = 0;
      bool done = !active;

      SimplePoint<Tt, Tv>* myRowPts = smemRowPts + threadIdx.x * kStride;
      size_t nChunks = (maxFsz + ChunkSize - 1) / ChunkSize;

      // --- Chunked shared-memory processing ---
      for (size_t chunk = 0; chunk < nChunks; ++chunk)
      {
        size_t chunkStart = chunk * ChunkSize;

        // Load body + 1 lookahead entry for this row's chunk
        size_t nLoad = 0;
        if (chunkStart < fsz)
        {
          size_t loadEnd = chunkStart + ChunkSize + 1;
          if (loadEnd > fsz) loadEnd = fsz;
          nLoad = loadEnd - chunkStart;
        }

        // Column threads cooperatively load this row's chunk
        for (size_t k = threadIdx.y; k < nLoad; k += blockDim.y)
        {
          myRowPts[k] = params.rowPoints[fOffset + chunkStart + k];
        }
        __syncthreads();

        // Continue merge scan using shared memory for row breakpoints
        if (!done)
        {
          size_t fiChunkEnd = chunkStart + ChunkSize;
          if (fiChunkEnd > fsz) fiChunkEnd = fsz;

          while (t < b && fi < fiChunkEnd)
          {
            Tt tprev = t;
            Tv fv = myRowPts[fi - chunkStart].v;
            Tv gv = gpts[gi].v;

            if (fi + 1 < fsz && gi + 1 < gsz)
            {
              auto delta = myRowPts[fi + 1 - chunkStart].t - gpts[gi + 1].t;
              if (delta <= 0)
              {
                ++fi;
              }
              if (delta >= 0)
              {
                ++gi;
              }
            }
            else
            {
              if (fi + 1 < fsz)
              {
                ++fi;
              }
              else if (gi + 1 < gsz)
              {
                ++gi;
              }
              else
              {
                ret += (b - tprev) * op(fv, gv);
                done = true;
                break;
              }
            }

            if (!done)
            {
              t = max(myRowPts[fi - chunkStart].t, gpts[gi].t);
              ret += (t - tprev) * op(fv, gv);
            }
          }
        }

        __syncthreads();
      }

      if (active)
      {
        params.matrix[iLocal * params.nCols + jLocal] = op(ret);
      }
    }

    /// Host-side launch wrapper.
    template <typename Tt, typename Tv, typename ComboOp>
    void launch_pcf_block_integrate(dim3 gridDim, dim3 blockDim, const PcfBlockKernelParams<Tt, Tv>& params, Tt a, Tt b, ComboOp op)
    {
      constexpr size_t CS = PCF_SMEM_CHUNK_SIZE;
      size_t smemBytes =
          blockDim.x * (CS + 1) * sizeof(SimplePoint<Tt, Tv>) +
          (blockDim.x + 1) * sizeof(size_t);
      cuda_pcf_block_integrate_smem<CS, Tt, Tv, ComboOp>
          <<<gridDim, blockDim, smemBytes>>>(params, a, b, op);
      CHK_CUDA(cudaGetLastError());
    }

  } // namespace internal

} // namespace mpcf

#endif
