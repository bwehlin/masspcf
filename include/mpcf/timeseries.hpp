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

#include "functional/pcf.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>

namespace mpcf
{
  template <typename Tt, typename Tv = Tt>
  class TimeSeries
  {
  public:
    using time_type = Tt;
    using value_type = Tv;

    TimeSeries()
      : m_start_time(0), m_time_step(1)
    { }

    explicit TimeSeries(Pcf<Tt, Tv> pcf, Tt start_time = 0, Tt time_step = 1)
      : m_pcf(std::move(pcf)), m_start_time(start_time), m_time_step(time_step)
    {
      if (m_time_step <= 0)
      {
        throw std::invalid_argument("time_step must be positive");
      }
    }

    /// Construct from chrono durations -- converts to float seconds.
    template <typename Rep1, typename Period1, typename Rep2, typename Period2>
    TimeSeries(Pcf<Tt, Tv> pcf,
               std::chrono::duration<Rep1, Period1> start,
               std::chrono::duration<Rep2, Period2> step)
      : m_pcf(std::move(pcf)),
        m_start_time(std::chrono::duration<Tt>(start).count()),
        m_time_step(std::chrono::duration<Tt>(step).count())
    {
      if (m_time_step <= 0)
      {
        throw std::invalid_argument("time_step must be positive");
      }
    }

    [[nodiscard]] value_type evaluate(time_type real_t) const
    {
      time_type pcf_t = (real_t - m_start_time) / m_time_step;
      if (pcf_t < 0)
        return std::numeric_limits<value_type>::quiet_NaN();
      if (m_pcf.size() > 0 && pcf_t > m_pcf.points().back().t)
        return std::numeric_limits<value_type>::quiet_NaN();
      return m_pcf.evaluate(pcf_t);
    }

    template <typename TIn, typename TOut>
    void evaluate(const TIn& sorted_real_times, TOut& out, size_t n) const
    {
      for (size_t i = 0; i < n; ++i)
      {
        time_type real_t = sorted_real_times(std::vector<size_t>{i});
        out(std::vector<size_t>{i}) = evaluate(real_t);
      }
    }

    /// Evaluate at a chrono time point.
    /// Reconstructs start and step in the query's tick unit, subtracts
    /// in integer chrono space (exact), then divides in float.
    template <typename Rep, typename Period>
    [[nodiscard]] value_type evaluate(
        std::chrono::duration<Rep, Period> query) const
    {
      using D = std::chrono::duration<Rep, Period>;
      // Cast float seconds -> integer ticks in the query's unit
      auto start = std::chrono::duration_cast<D>(
          std::chrono::duration<Tt>(m_start_time));
      auto step = std::chrono::duration_cast<D>(
          std::chrono::duration<Tt>(m_time_step));
      // Integer subtraction -- exact, result is small
      Rep offset = (query - start).count();
      Rep step_count = step.count();
      if (step_count == 0)
      {
        // Query unit too coarse for this time_step; fall back to float
        return evaluate(std::chrono::duration<Tt>(query).count());
      }
      // Small integer -> float division
      time_type pcf_t = static_cast<time_type>(offset)
                      / static_cast<time_type>(step_count);
      if (pcf_t < 0)
        return std::numeric_limits<value_type>::quiet_NaN();
      if (m_pcf.size() > 0 && pcf_t > m_pcf.points().back().t)
        return std::numeric_limits<value_type>::quiet_NaN();
      return m_pcf.evaluate(pcf_t);
    }

    [[nodiscard]] const Pcf<Tt, Tv>& pcf() const noexcept { return m_pcf; }
    [[nodiscard]] Tt start_time() const noexcept { return m_start_time; }
    [[nodiscard]] Tt time_step() const noexcept { return m_time_step; }

    [[nodiscard]] Tt end_time() const noexcept
    {
      if (m_pcf.size() == 0)
        return m_start_time;
      return m_start_time + m_pcf.points().back().t * m_time_step;
    }

    [[nodiscard]] size_t size() const noexcept { return m_pcf.size(); }

    bool operator==(const TimeSeries& rhs) const
    {
      return m_pcf == rhs.m_pcf && m_start_time == rhs.m_start_time && m_time_step == rhs.m_time_step;
    }

    bool operator!=(const TimeSeries& rhs) const
    {
      return !(*this == rhs);
    }

    [[nodiscard]] std::string to_string() const
    {
      std::stringstream ss;
      ss << "TimeSeries(start_time=" << m_start_time << ", time_step=" << m_time_step
         << ", pcf=" << m_pcf.to_string() << ")";
      return ss.str();
    }

  private:
    Pcf<Tt, Tv> m_pcf;
    Tt m_start_time;
    Tt m_time_step;
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
