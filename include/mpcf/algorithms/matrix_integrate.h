/*
* Copyright 2024 Bjorn Wehlin
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

#include "iterate_rectangles.h"
#include "../pcf.h"
#include "../executor.h"
#include "../task.h"

#ifdef BUILD_WITH_CUDA
#include "../cuda/cuda_matrix_integrate.cuh"
#endif

#include <vector>

#include <taskflow/algorithm/for_each.hpp>

namespace mpcf
{
  template <typename Tt, typename Tv, typename RectangleOp>
  Tv integrate(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g, RectangleOp op, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;

    Tv val = 0.f;
    iterate_rectangles(f.points(), g.points(), [&val, &op](const rect_t& rect) -> void {
      val += (rect.right - rect.left) * op(rect);
    }, a, b);
    
    return val;
  }
  
  template <typename Tv>
  void make_lower_triangle(Executor& exec, Tv* out, size_t sz)
  {
    tf::Taskflow flow;
    flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [out, sz](size_t i) {
      for (size_t j = 0; j < i; ++j)
      {
        out[i * sz + j] = out[j * sz + i];
      }
    });
    auto future = exec.cpu()->run(flow);
    future.wait();
  }
  
  template <typename Tt, typename Tv, typename RectangleOp>
  void matrix_integrate(Executor& exec, Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, const RectangleOp& op, bool symmetric = true, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    auto sz = fs.size();
    tf::Taskflow flow;
    
    flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [out, &fs, &op, symmetric, a, b, sz](size_t i) {
      auto j = symmetric ? i : 0ul;
      for (; j < sz; ++j)
      {
        out[i * sz + j] = integrate<Tt, Tv, RectangleOp>(fs[i], fs[j], op, a, b);
      }
    });

    auto future = exec.cpu()->run(std::move(flow));
    future.wait();
  }
  
  template <typename Tt, typename Tv, typename RectangleOp>
  void matrix_integrate(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, const RectangleOp& op, bool symmetric = false, Tt a = 0.f, Tt b = std::numeric_limits<Tt>::max())
  {
    matrix_integrate<Tt, Tv, RectangleOp>(default_executor(), out, fs, op, symmetric, a, b);
  }
  
  
  template <typename Tt, typename Tv, typename TOperation>
  class MatrixIntegrateCpuTask : public mpcf::StoppableTask<void>
  {
  public:
    MatrixIntegrateCpuTask(Tv* out, std::vector<Pcf<Tt, Tv>>&& fs, TOperation op)
      : m_fs(std::move(fs))
      , m_out(out)
      , m_op(op)
    { }
    
  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto sz = m_fs.size();
      auto totalWorkPerStep = (sz * (sz - 1)) / 2;

      next_step(totalWorkPerStep, "Computing upper triangle.", "integral");

      tf::Taskflow flow;
      std::vector<tf::Task> tasks;
      
      tasks.emplace_back(flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [this](size_t i) {
        if (stop_requested())
        {
          return;
        }
        compute_row(i);
      }));

      
      tasks.emplace_back(flow.emplace([this, totalWorkPerStep] {
        next_step(totalWorkPerStep, "Filling in lower triangle.", "element");
      }));

      tasks.emplace_back(flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [this](size_t i) {
        if (stop_requested())
        {
          return;
        }
        symmetrize_row(i);
      }));

      tasks.emplace_back(create_terminal_task(flow));
      flow.linearize(tasks);

      return exec.cpu()->run(std::move(flow));
    }
    
    void compute_row(size_t i)
    {
      auto sz = m_fs.size();
      for (size_t j = i + 1; j < sz; ++j)
      {
        m_out[i * sz + j] = m_op(integrate<Tt, Tv>(m_fs[i], m_fs[j], [this](const Rectangle<Tt, Tv>& rect){ 
                                                                return m_op(rect.top, rect.bottom); 
                                                              }, 0, std::numeric_limits<Tt>::max()));
      }
      add_progress(sz - i - 1);
    }

    void symmetrize_row(size_t i)
    {
      auto sz = m_fs.size();
      for (size_t j = 0; j < i; ++j)
      {
        m_out[i * sz + j] = m_out[j * sz + i];
      }
      add_progress(sz - i - 1);
    }
    
    std::vector<Pcf<Tt, Tv>> m_fs;
    Tv* m_out;
    TOperation m_op;
  };
  
  template <typename Tt, typename Tv>
  void matrix_l1_dist(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, Hardware hardware = Hardware::CPU, Executor& executor = default_executor())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;
    
    switch (hardware)
    {
#ifdef BUILD_WITH_CUDA
    case Hardware::CUDA:
      cuda_matrix_l1_dist<Tt, Tv>(out, fs);
      break;
#endif
    default:
      matrix_integrate(executor, out, fs, [](const rect_t& rect) -> Tv {
        return std::abs(rect.top - rect.bottom);
      }, true);
    }
    
    make_lower_triangle<Tv>(executor, out, fs.size());
  }
  
  template <typename Tt, typename Tv, typename TOperation>
  void integrate_matrix(Tv* out, const std::vector<Pcf<Tt, Tv>>& fs, TOperation op, Tv a = Tv(0), Tv b = std::numeric_limits<Tv>::max(), Hardware hardware = Hardware::CPU, Executor& executor = default_executor())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;
    
    switch (hardware)
    {
#ifdef BUILD_WITH_CUDA
    case Hardware::CUDA:
      cuda_matrix_integrate<Tt, Tv, TOperation>(out, fs, op, a, b, executor);
      break;
#endif
    default:
      // TODO
      matrix_integrate(executor, out, fs, [](const rect_t& rect) -> Tv {
        return std::abs(rect.top - rect.bottom);
      }, true);
    }
    
    make_lower_triangle<Tv>(executor, out, fs.size());
  }
}

#endif
