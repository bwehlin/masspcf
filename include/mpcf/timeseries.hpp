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
#include "timeseries/interpolation.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace mpcf
{
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
      if constexpr (std::is_arithmetic_v<Tv>)
      {
        m_times.push_back(Tt(0));
        m_values.push_back(Tv(0));
      }
      else
      {
        m_times.push_back(Tt(0));
        m_values.emplace_back();
      }
    }

    /// Construct from times, values (flattened row-major), and n_channels.
    TimeSeries(std::vector<Tt> times, std::vector<Tv> values,
               size_t n_channels, Tt start_time, Tt time_step,
               InterpolationMode interpolation = InterpolationMode::Nearest)
      : m_times(std::move(times)), m_values(std::move(values)),
        m_n_channels(n_channels),
        m_start_time(start_time), m_time_step(time_step)
    {
      validate_construction();
      set_interpolation(interpolation);
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
        m_time_step(std::chrono::duration<Tt>(step).count())
    {
      validate_construction();
      set_interpolation(interpolation);
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

    /// Current interpolation mode when the variant holds a built-in tag.
    /// For a custom strategy, this returns `InterpolationMode::Nearest`
    /// (not meaningful); check `has_custom_strategy()` first.
    [[nodiscard]] InterpolationMode interpolation() const noexcept
    {
      return to_mode(m_interp);
    }

    /// Set the built-in interpolation mode. Throws if `mode` is `Linear`
    /// and `Tv` does not satisfy `LinearlyBlendable<Tt, Tv>` (e.g., Barcode).
    void set_interpolation(InterpolationMode mode)
    {
      if (mode == InterpolationMode::Linear)
      {
        if constexpr (LinearlyBlendable<Tt, Tv>)
        {
          m_interp = LinearTag{};
          return;
        }
        else
        {
          throw std::invalid_argument(
              "linear interpolation is not defined for this value type");
        }
      }
      m_interp = NearestTag{};
    }

    /// Attach a custom interpolation strategy (shared, so the same
    /// strategy instance can be reused across multiple TimeSeries). Pass
    /// a null pointer to revert to nearest.
    void set_strategy(std::shared_ptr<InterpolationStrategy<Tt, Tv>> strategy)
    {
      if (!strategy)
      {
        m_interp = NearestTag{};
        return;
      }
      m_interp = CustomStrategy<Tt, Tv>{std::move(strategy)};
    }

    [[nodiscard]] std::shared_ptr<InterpolationStrategy<Tt, Tv>>
    strategy() const noexcept
    {
      if (auto* c = std::get_if<CustomStrategy<Tt, Tv>>(&m_interp))
        return c->ptr;
      return {};
    }

    [[nodiscard]] bool has_custom_strategy() const noexcept
    {
      return holds_custom_strategy(m_interp);
    }

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
      if (m_times != rhs.m_times || m_values != rhs.m_values
          || m_n_channels != rhs.m_n_channels
          || m_start_time != rhs.m_start_time
          || m_time_step != rhs.m_time_step)
        return false;
      // Compare by interpolation mode; custom strategies are equal only
      // when they share the same shared_ptr instance.
      if (has_custom_strategy() != rhs.has_custom_strategy())
        return false;
      if (has_custom_strategy())
        return strategy().get() == rhs.strategy().get();
      return to_mode(m_interp) == to_mode(rhs.m_interp);
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
      if (has_custom_strategy())
        ss << ", interpolation=custom";
      else if (to_mode(m_interp) == InterpolationMode::Linear)
        ss << ", interpolation=linear";
      ss << ")";
      return ss.str();
    }

  private:
    void validate_construction()
    {
      if (m_time_step <= 0)
        throw std::invalid_argument("time_step must be positive");
      if (m_values.size() != m_times.size() * m_n_channels)
        throw std::invalid_argument(
            "values size must equal times size * n_channels");
      if constexpr (!std::is_arithmetic_v<Tv>)
      {
        if (m_n_channels != 1)
          throw std::invalid_argument(
              "multi-channel is only supported for arithmetic value types");
      }
    }

    /// Find the interval index for a pcf time (binary search on m_times).
    [[nodiscard]] size_t find_interval(time_type pcf_t) const
    {
      auto it = std::upper_bound(m_times.begin(), m_times.end(), pcf_t);
      if (it == m_times.begin())
        return 0; // before first breakpoint
      --it;
      return static_cast<size_t>(it - m_times.begin());
    }

    /// Snap pcf_t to nearby breakpoints (domain boundaries and the bracketing
    /// pair) within the given relative tolerance. Mirrors the pre-refactor
    /// inline snapping logic.
    void snap_in_place(time_type& pcf_t, size_t& idx, Tt snap_tol) const
    {
      if (snap_tol <= 0 || m_times.empty())
        return;

      auto near = [snap_tol](Tt a, Tt b) {
        return std::abs(a - b)
            <= snap_tol * std::max({Tt(1), std::abs(a), std::abs(b)});
      };

      if (pcf_t < m_times.front() && near(pcf_t, m_times.front()))
        pcf_t = m_times.front();
      else if (pcf_t > m_times.back() && near(pcf_t, m_times.back()))
        pcf_t = m_times.back();

      // Re-find interval in case the boundary snap changed domain membership
      if (pcf_t < 0 || pcf_t > m_times.back())
        return;
      idx = find_interval(pcf_t);

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

      Tt diff = pcf_t - m_times[idx];
      Tt ref = std::max({Tt(1), std::abs(pcf_t), std::abs(m_times[idx])});
      if (diff > 0 && diff <= snap_tol * ref)
        pcf_t = m_times[idx];
    }

  public:
    /// Batched evaluation at a span of pcf-time queries. Returns one Tv per
    /// (query, channel). Queries outside the domain get OutOfDomainValue.
    ///
    /// Evaluation path:
    ///   1) For each query, snap + binary-search to the bracketing interval.
    ///   2) Delegate interpolation to m_strategy (creating the default
    ///      strategy for m_interpolation on first use).
    ///
    /// Multi-channel: for scalar Tv with n_channels > 1, the same bracketing
    /// interval is used for all channels; the strategy is called once per
    /// channel with length-n spans for that channel.
    [[nodiscard]] std::vector<Tv> evaluate_batch(
        std::span<const Tt> pcf_queries,
        Tt snap_tol = default_snap_tol()) const
    {
      const size_t n = pcf_queries.size();
      std::vector<Tv> result(n * m_n_channels);
      if (n == 0)
        return result;

      // Per-query bracketing: snap + find interval once, store idx.
      // Out-of-domain queries are marked with idx == SIZE_MAX.
      constexpr size_t OUT_OF_DOMAIN = static_cast<size_t>(-1);
      std::vector<Tt> qs(n);
      std::vector<Tt> t_lefts(n);
      std::vector<Tt> t_rights(n);
      std::vector<size_t> idxs(n, OUT_OF_DOMAIN);

      for (size_t i = 0; i < n; ++i)
      {
        Tt q = pcf_queries[i];
        size_t idx = 0;
        if (!m_times.empty())
          idx = find_interval(q);
        snap_in_place(q, idx, snap_tol);

        if (m_times.empty() || q < 0 || q > m_times.back())
        {
          qs[i] = q;
          t_lefts[i] = Tt(0);
          t_rights[i] = Tt(0);
          continue;
        }

        qs[i] = q;
        idxs[i] = idx;
        t_lefts[i] = m_times[idx];
        t_rights[i] = (idx + 1 < m_times.size())
            ? m_times[idx + 1]
            : m_times[idx];  // right boundary: left == right
      }

      // Per-channel interpolation via std::visit on the variant.
      // Nearest / Linear are inlined; custom strategy dispatches through
      // its virtual evaluate.
      std::vector<Tv> v_lefts(n);
      std::vector<Tv> v_rights(n);

      for (size_t c = 0; c < m_n_channels; ++c)
      {
        for (size_t i = 0; i < n; ++i)
        {
          if (idxs[i] == OUT_OF_DOMAIN)
            continue;
          size_t idx = idxs[i];
          v_lefts[i] = m_values[(idx * m_n_channels) + c];
          v_rights[i] = (idx + 1 < m_times.size())
              ? m_values[((idx + 1) * m_n_channels) + c]
              : v_lefts[i];
        }

        std::visit([&](const auto& tag) {
          using Mode = std::decay_t<decltype(tag)>;
          if constexpr (std::is_same_v<Mode, NearestTag>)
          {
            for (size_t i = 0; i < n; ++i)
              result[(i * m_n_channels) + c] = (idxs[i] == OUT_OF_DOMAIN)
                  ? OutOfDomainValue<Tv>::get()
                  : v_lefts[i];
          }
          else if constexpr (std::is_same_v<Mode, LinearTag>)
          {
            if constexpr (LinearlyBlendable<Tt, Tv>)
            {
              for (size_t i = 0; i < n; ++i)
              {
                if (idxs[i] == OUT_OF_DOMAIN)
                {
                  result[(i * m_n_channels) + c] = OutOfDomainValue<Tv>::get();
                  continue;
                }
                Tt dt = t_rights[i] - t_lefts[i];
                if (dt <= Tt(0))
                {
                  result[(i * m_n_channels) + c] = v_lefts[i];
                  continue;
                }
                Tt alpha = (qs[i] - t_lefts[i]) / dt;
                result[(i * m_n_channels) + c] = static_cast<Tv>(
                    ((Tt(1) - alpha) * v_lefts[i]) + (alpha * v_rights[i]));
              }
            }
            else
            {
              // Should be unreachable: set_interpolation(Linear) throws
              // for non-blendable Tv.
              throw std::logic_error(
                  "LinearTag set for non-blendable value type");
            }
          }
          else
          {
            auto out = tag.ptr->evaluate(
                qs, t_lefts, t_rights, v_lefts, v_rights);
            if (out.size() != n)
              throw std::runtime_error(
                  "InterpolationStrategy.evaluate returned wrong-length vector");
            for (size_t i = 0; i < n; ++i)
              result[(i * m_n_channels) + c] = (idxs[i] == OUT_OF_DOMAIN)
                  ? OutOfDomainValue<Tv>::get()
                  : out[i];
          }
        }, m_interp);
      }
      return result;
    }

  private:
    [[nodiscard]] Tensor<Tv> evaluate_at_pcf_t(time_type pcf_t,
                                               Tt snap_tol = default_snap_tol()) const
    {
      Tt query_buf[1] = { pcf_t };
      auto vals = evaluate_batch(
          std::span<const Tt>(query_buf, 1), snap_tol);

      Tensor<Tv> result(std::vector<size_t>{m_n_channels});
      for (size_t c = 0; c < m_n_channels; ++c)
        result(std::vector<size_t>{c}) = vals[c];
      return result;
    }

    std::vector<Tt> m_times;
    std::vector<Tv> m_values;  // row-major (n_times, n_channels)
    size_t m_n_channels;
    Tt m_start_time;
    Tt m_time_step;
    InterpolationChoice<Tt, Tv> m_interp = NearestTag{};
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
