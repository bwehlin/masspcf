/*
* Copyright 2026 Bjorn Wehlin
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

#ifndef MPCF_ALGORITHMS_EMBED_TIME_DELAY_HPP
#define MPCF_ALGORITHMS_EMBED_TIME_DELAY_HPP

#include "../timeseries.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mpcf
{
  namespace detail
  {
    /// Embed a single TimeSeries over the base time range [valid_start, valid_end].
    /// Only breakpoint real times within this range are used as base times.
    template <typename Tt, typename Tv>
    PointCloud<Tv> embed_time_delay_range(
        const TimeSeries<Tt, Tv>& ts, size_t dimension, Tt delay,
        Tt valid_start, Tt valid_end)
    {
      auto const& internal_times = ts.times();
      auto start = ts.start_time();
      auto step = ts.time_step();
      auto nc = ts.n_channels();
      size_t out_cols = dimension * nc;

      // Collect base times (breakpoint real times in [valid_start, valid_end])
      std::vector<Tt> base_times;
      base_times.reserve(internal_times.size());
      for (size_t i = 0; i < internal_times.size(); ++i)
      {
        Tt real_t = start + internal_times[i] * step;
        if (real_t >= valid_start && real_t <= valid_end)
          base_times.push_back(real_t);
      }

      if (base_times.empty())
      {
        return PointCloud<Tv>(std::vector<size_t>{0, out_cols});
      }

      size_t n_out = base_times.size();
      PointCloud<Tv> cloud(std::vector<size_t>{n_out, out_cols});

      for (size_t i = 0; i < n_out; ++i)
      {
        Tt t = base_times[i];
        for (size_t j = 0; j < dimension; ++j)
        {
          // j=0 is the oldest (t - (d-1)*delay), j=d-1 is current (t)
          Tt eval_t = t - static_cast<Tt>(dimension - 1 - j) * delay;
          auto vals = ts.evaluate(eval_t);
          for (size_t ch = 0; ch < nc; ++ch)
          {
            cloud(std::vector<size_t>{i, j * nc + ch}) =
                vals(std::vector<size_t>{ch});
          }
        }
      }

      return cloud;
    }

    /// Compute backward-anchored window boundaries.
    /// Returns window start times in chronological order.
    /// The last window ends at valid_end, first window may be partial.
    template <typename Tt>
    std::vector<std::pair<Tt, Tt>> compute_windows(
        Tt valid_start, Tt valid_end, Tt window, Tt stride)
    {
      std::vector<std::pair<Tt, Tt>> windows;

      // Work backward from valid_end
      Tt win_end = valid_end;
      while (win_end > valid_start)
      {
        Tt win_start = win_end - window;
        if (win_start < valid_start)
          win_start = valid_start;
        windows.push_back({win_start, win_end});
        if (win_start <= valid_start)
          break;  // First window reached valid_start, done
        win_end -= stride;
      }

      // Reverse to chronological order
      std::reverse(windows.begin(), windows.end());
      return windows;
    }
  }

  /// Time delay embedding of a single TimeSeries.
  ///
  /// Produces a tensor of point clouds (one per window, or one if no windowing).
  /// Each point cloud has shape (n_points, dimension * n_channels).
  /// Embedding vectors look backward: [x(t-(d-1)*tau), ..., x(t-tau), x(t)].
  template <typename Tt, typename Tv>
  Tensor<PointCloud<Tv>> embed_time_delay(
      const TimeSeries<Tt, Tv>& ts, size_t dimension, Tt delay,
      Tt window = Tt(0), Tt stride = Tt(0))
  {
    if (dimension < 1)
      throw std::invalid_argument("dimension must be >= 1");
    if (delay <= 0)
      throw std::invalid_argument("delay must be positive");

    Tt valid_start = ts.start_time()
                   + static_cast<Tt>(dimension - 1) * delay;
    Tt valid_end = ts.end_time();

    if (valid_start > valid_end)
      throw std::invalid_argument(
          "time series too short for the given dimension and delay");

    if (window <= Tt(0))
    {
      // No windowing: single point cloud
      auto cloud = detail::embed_time_delay_range(
          ts, dimension, delay, valid_start, valid_end);
      Tensor<PointCloud<Tv>> result(std::vector<size_t>{1});
      result(std::vector<size_t>{0}) = std::move(cloud);
      return result;
    }

    if (stride <= Tt(0))
      stride = window;

    auto windows = detail::compute_windows(
        valid_start, valid_end, window, stride);

    Tensor<PointCloud<Tv>> result(
        std::vector<size_t>{windows.size()});
    for (size_t i = 0; i < windows.size(); ++i)
    {
      auto [ws, we] = windows[i];
      result(std::vector<size_t>{i}) =
          detail::embed_time_delay_range(
              ts, dimension, delay, ws, we);
    }
    return result;
  }

  /// Time delay embedding of a tensor of TimeSeries.
  ///
  /// Computes a common time domain [max(start), min(end)] across all series.
  /// Output shape: input_shape (no windowing) or input_shape + (n_windows,).
  template <typename Tt, typename Tv>
  Tensor<PointCloud<Tv>> embed_time_delay(
      const Tensor<TimeSeries<Tt, Tv>>& ts_tensor,
      size_t dimension, Tt delay,
      Tt window = Tt(0), Tt stride = Tt(0))
  {
    if (dimension < 1)
      throw std::invalid_argument("dimension must be >= 1");
    if (delay <= 0)
      throw std::invalid_argument("delay must be positive");

    // Compute common domain
    Tt common_start = -std::numeric_limits<Tt>::infinity();
    Tt common_end = std::numeric_limits<Tt>::infinity();

    walk(ts_tensor, [&](const std::vector<size_t>& idx) {
      const auto& ts = ts_tensor(idx);
      common_start = std::max(common_start, ts.start_time());
      common_end = std::min(common_end, ts.end_time());
    });

    Tt valid_start = common_start
                   + static_cast<Tt>(dimension - 1) * delay;
    Tt valid_end = common_end;

    if (valid_start > valid_end)
      throw std::invalid_argument(
          "common time domain too short for the given dimension and delay");

    if (stride <= Tt(0))
      stride = window;

    if (window <= Tt(0))
    {
      // No windowing: output shape = input_shape
      Tensor<PointCloud<Tv>> result(ts_tensor.shape());
      walk(ts_tensor, [&](const std::vector<size_t>& idx) {
        result(idx) = detail::embed_time_delay_range(
            ts_tensor(idx), dimension, delay,
            valid_start, valid_end);
      });
      return result;
    }

    // Windowed: output shape = input_shape + (n_windows,)
    auto windows = detail::compute_windows(
        valid_start, valid_end, window, stride);
    size_t n_windows = windows.size();

    auto out_shape = ts_tensor.shape();
    std::vector<size_t> out_shape_vec(
        out_shape.begin(), out_shape.end());
    out_shape_vec.push_back(n_windows);

    Tensor<PointCloud<Tv>> result(out_shape_vec);

    walk(ts_tensor, [&](const std::vector<size_t>& idx) {
      for (size_t w = 0; w < n_windows; ++w)
      {
        auto [ws, we] = windows[w];
        auto out_idx = idx;
        out_idx.push_back(w);
        result(out_idx) = detail::embed_time_delay_range(
            ts_tensor(idx), dimension, delay, ws, we);
      }
    });

    return result;
  }
}

#endif // MPCF_ALGORITHMS_EMBED_TIME_DELAY_HPP
