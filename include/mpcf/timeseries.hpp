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

#ifndef MPCF_TIMESERIES_H
#define MPCF_TIMESERIES_H

#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <vector>

namespace mpcf
{
  enum class InterpolationMode : uint8_t { Nearest = 0, Linear = 1 };

  template <typename Tt, typename Tv = Tt>
  class TimeSeries
  {
  public:
    using time_type = Tt;
    using value_type = Tv;

    static constexpr Tt default_snap_tol()
    {
      return sizeof(Tt) <= 4 ? Tt(1e-5) : Tt(1e-9);
    }

    TimeSeries()
      : m_n_channels(1), m_start_time(0), m_time_step(1)
    {
      m_times.push_back(Tt(0));
      m_values.push_back(Tv(0));
    }

    /// Construct from times, values (flattened row-major), and n_channels.
    TimeSeries(std::vector<Tt> times, std::vector<Tv> values,
               size_t n_channels, Tt start_time, Tt time_step,
               InterpolationMode interpolation = InterpolationMode::Nearest)
      : m_times(std::move(times)), m_values(std::move(values)),
        m_n_channels(n_channels),
        m_start_time(start_time), m_time_step(time_step),
        m_interpolation(interpolation)
    {
      if (m_time_step <= 0)
        throw std::invalid_argument("time_step must be positive");
      if (m_values.size() != m_times.size() * m_n_channels)
        throw std::invalid_argument("values size must equal times size * n_channels");
    }

    /// Construct from chrono durations.
    template <typename Rep1, typename Period1, typename Rep2, typename Period2>
    TimeSeries(std::vector<Tt> times, std::vector<Tv> values,
               size_t n_channels,
               std::chrono::duration<Rep1, Period1> start,
               std::chrono::duration<Rep2, Period2> step,
               InterpolationMode interpolation = InterpolationMode::Nearest)
      : m_times(std::move(times)), m_values(std::move(values)),
        m_n_channels(n_channels),
        m_start_time(std::chrono::duration<Tt>(start).count()),
        m_time_step(std::chrono::duration<Tt>(step).count()),
        m_interpolation(interpolation)
    {
      if (m_time_step <= 0)
        throw std::invalid_argument("time_step must be positive");
      if (m_values.size() != m_times.size() * m_n_channels)
        throw std::invalid_argument("values size must equal times size * n_channels");
    }

    /// Evaluate all channels at a real time. Returns Tensor<Tv> of shape (n_channels,).
    /// snap_tol is the relative error tolerance for snapping to breakpoints.
    [[nodiscard]] Tensor<Tv> evaluate(time_type real_t,
                                      Tt snap_tol = default_snap_tol()) const
    {
      time_type pcf_t = (real_t - m_start_time) / m_time_step;
      return evaluate_at_pcf_t(pcf_t, snap_tol);
    }

    /// Batch evaluate (generic container interface for tensor_eval).
    template <typename TIn, typename TOut>
    void evaluate(const TIn& sorted_real_times, TOut& out, size_t n,
                  Tt snap_tol = default_snap_tol()) const
    {
      for (size_t i = 0; i < n; ++i)
      {
        auto real_t = sorted_real_times(std::vector<size_t>{i});
        out(std::vector<size_t>{i}) = evaluate(real_t, snap_tol);
      }
    }

    /// Evaluate at a chrono time point.
    template <typename Rep, typename Period>
    [[nodiscard]] Tensor<Tv> evaluate(
        std::chrono::duration<Rep, Period> query,
        Tt snap_tol = default_snap_tol()) const
    {
      using D = std::chrono::duration<Rep, Period>;
      auto start = std::chrono::duration_cast<D>(
          std::chrono::duration<Tt>(m_start_time));
      auto step = std::chrono::duration_cast<D>(
          std::chrono::duration<Tt>(m_time_step));
      Rep offset = (query - start).count();
      Rep step_count = step.count();
      if (step_count == 0)
      {
        return evaluate(std::chrono::duration<Tt>(query).count(), snap_tol);
      }
      time_type pcf_t = static_cast<time_type>(offset)
                      / static_cast<time_type>(step_count);
      return evaluate_at_pcf_t(pcf_t, snap_tol);
    }

    [[nodiscard]] InterpolationMode interpolation() const noexcept { return m_interpolation; }
    void set_interpolation(InterpolationMode mode) noexcept { m_interpolation = mode; }

    [[nodiscard]] size_t n_channels() const noexcept { return m_n_channels; }
    [[nodiscard]] size_t n_times() const noexcept { return m_times.size(); }
    [[nodiscard]] const std::vector<Tt>& times() const noexcept { return m_times; }
    [[nodiscard]] const std::vector<Tv>& values() const noexcept { return m_values; }
    [[nodiscard]] Tt start_time() const noexcept { return m_start_time; }
    [[nodiscard]] Tt time_step() const noexcept { return m_time_step; }

    [[nodiscard]] Tv value(size_t time_idx, size_t channel) const
    {
      return m_values[time_idx * m_n_channels + channel];
    }

    [[nodiscard]] Tt end_time() const noexcept
    {
      if (m_times.empty())
        return m_start_time;
      return m_start_time + m_times.back() * m_time_step;
    }

    bool operator==(const TimeSeries& rhs) const
    {
      return m_times == rhs.m_times && m_values == rhs.m_values
          && m_n_channels == rhs.m_n_channels
          && m_start_time == rhs.m_start_time
          && m_time_step == rhs.m_time_step
          && m_interpolation == rhs.m_interpolation;
    }

    bool operator!=(const TimeSeries& rhs) const
    {
      return !(*this == rhs);
    }

    [[nodiscard]] std::string to_string() const
    {
      std::stringstream ss;
      ss << "TimeSeries(start_time=" << m_start_time
         << ", n_times=" << m_times.size()
         << ", n_channels=" << m_n_channels;
      if (m_interpolation != InterpolationMode::Nearest)
        ss << ", interpolation=linear";
      ss << ")";
      return ss.str();
    }

  private:
    /// Find the interval index for a pcf time (binary search on m_times).
    [[nodiscard]] size_t find_interval(time_type pcf_t) const
    {
      auto it = std::upper_bound(m_times.begin(), m_times.end(), pcf_t);
      if (it == m_times.begin())
        return 0; // before first breakpoint
      --it;
      return static_cast<size_t>(it - m_times.begin());
    }

    [[nodiscard]] Tensor<Tv> evaluate_at_pcf_t(time_type pcf_t,
                                               Tt snap_tol = default_snap_tol()) const
    {
      Tensor<Tv> result(std::vector<size_t>{m_n_channels});

      // Snap to domain boundaries if within relative error
      if (snap_tol > 0 && !m_times.empty())
      {
        auto near = [snap_tol](Tt a, Tt b) {
          return std::abs(a - b)
              <= snap_tol * std::max({Tt(1), std::abs(a), std::abs(b)});
        };

        if (pcf_t < m_times.front() && near(pcf_t, m_times.front()))
          pcf_t = m_times.front();
        else if (pcf_t > m_times.back() && near(pcf_t, m_times.back()))
          pcf_t = m_times.back();
      }

      if (pcf_t < 0 || (m_times.size() > 0 && pcf_t > m_times.back()))
      {
        for (size_t c = 0; c < m_n_channels; ++c)
          result(std::vector<size_t>{c}) = std::numeric_limits<Tv>::quiet_NaN();
        return result;
      }

      size_t idx = find_interval(pcf_t);

      // Snap to nearest breakpoint if within relative error
      if (snap_tol > 0)
      {
        // Snap up to next breakpoint (drift left)
        if (idx + 1 < m_times.size())
        {
          Tt diff = m_times[idx + 1] - pcf_t;
          Tt ref = std::max({Tt(1), std::abs(pcf_t), std::abs(m_times[idx + 1])});
          if (diff > 0 && diff <= snap_tol * ref)
          {
            pcf_t = m_times[idx + 1];
            idx = idx + 1;
          }
        }

        // Snap down to current breakpoint (drift right)
        Tt diff = pcf_t - m_times[idx];
        Tt ref = std::max({Tt(1), std::abs(pcf_t), std::abs(m_times[idx])});
        if (diff > 0 && diff <= snap_tol * ref)
          pcf_t = m_times[idx];
      }

      if (m_interpolation == InterpolationMode::Linear
          && idx + 1 < m_times.size())
      {
        Tt alpha = (pcf_t - m_times[idx])
                 / (m_times[idx + 1] - m_times[idx]);
        for (size_t c = 0; c < m_n_channels; ++c)
        {
          Tv v0 = m_values[idx * m_n_channels + c];
          Tv v1 = m_values[(idx + 1) * m_n_channels + c];
          result(std::vector<size_t>{c}) =
              static_cast<Tv>((Tt(1) - alpha) * v0 + alpha * v1);
        }
      }
      else
      {
        for (size_t c = 0; c < m_n_channels; ++c)
          result(std::vector<size_t>{c}) = m_values[idx * m_n_channels + c];
      }
      return result;
    }

    std::vector<Tt> m_times;
    std::vector<Tv> m_values;  // row-major (n_times, n_channels)
    size_t m_n_channels;
    Tt m_start_time;
    Tt m_time_step;
    InterpolationMode m_interpolation = InterpolationMode::Nearest;
  };

  using TimeSeries_f32 = TimeSeries<float32_t, float32_t>;
  using TimeSeries_f64 = TimeSeries<float64_t, float64_t>;

  template <typename T>
  struct is_timeseries : std::false_type {};

  template <typename Tt, typename Tv>
  struct is_timeseries<TimeSeries<Tt, Tv>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_timeseries_v = is_timeseries<T>::value;

  template <typename Tt, typename Tv>
  void PrintTo(const TimeSeries<Tt, Tv>& ts, std::ostream* os)
  {
    *os << ts.to_string();
  }
}

#endif // MPCF_TIMESERIES_H
