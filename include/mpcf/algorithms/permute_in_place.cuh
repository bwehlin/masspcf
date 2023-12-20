#ifndef MPCF_ALGORITHM_PERMUTE_IN_PLACE_H
#define MPCF_ALGORITHM_PERMUTE_IN_PLACE_H

#include <vector>
#include <memory>

#include "../executor.h"
#include "subdivide.h"

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace mpcf
{
  /// Returns a cycle decomposition of the given permutation. It is assumed that
  /// if 'permutation' has size n, then all values 0,1,...,n-1 occur in 'permutation'
  inline std::vector<std::vector<size_t>> get_cycles(std::vector<size_t> permutation)
  {
    auto seen = permutation.size(); // Sentinel value
    std::vector<std::vector<size_t>> cycles;

    std::vector<size_t> cycle; // Kept outside the loop for performance
    size_t n = 0;
    size_t nextN = 0;
    for (size_t i = 0ul; i < permutation.size(); ++i)
    {
      if (permutation[i] == seen) 
      {
        // This item is already part of a cycle
        continue;
      }

      cycle.clear();
      cycle.emplace_back(i);
      n = permutation[i]; // n follows the start element 'i' around its orbit
      while (n != seen && n != i)
      {
        cycle.emplace_back(n);

        nextN = permutation[n];
        permutation[n] = seen;
        n = nextN;
      }

      if (cycle.size() > 1)
      {
        // We only care about nontrivial cycles
        cycles.emplace_back();
        cycles.back().swap(cycle);
      }
    }

    return cycles;
  }
  
  inline void invert_permutation(std::vector<std::vector<size_t>>& cycles)
  {
    for (auto & cycle : cycles)
    {
      std::reverse(cycle.begin(), cycle.end());
    }
  }
  
  template <typename RandomAccessIt>
  void apply_permutation(RandomAccessIt begin, const std::vector<std::vector<size_t>>& cycles)
  {
    using value_type = typename RandomAccessIt::value_type;
    auto tmp = *begin;
    for (auto const & cycle : cycles)
    {
      tmp = *(begin + cycle.back());
      for (auto cit = cycle.rbegin(); cit != std::prev(cycle.rend()); ++cit)
      {
        *(begin + *cit) = *(begin + *std::next(cit));
      }
      (begin + cycle.front())->swap(tmp);
    }
  }


  namespace detail
  {

    struct RowIndexer
    {
      size_t operator()(size_t row, size_t col, size_t n) const
      {
        return row * n + col;
      }
    };

    struct ColumnIndexer
    {
      size_t operator()(size_t col, size_t row, size_t n) const
      {
        return row * n + col;
      }
    };

    template <typename Tv, typename Indexer>
    inline void apply_matrix_reverse_cycle(Tv* matrix, size_t n, size_t jFirst, size_t jLast, const std::vector<size_t>& cycle, Indexer idx) 
    {
      // This code is written as if we are flipping rows of the matrix, but the indexer allows for flipping columns instead.

      std::vector<Tv> tmp(n);
      
      // The inverse of a cycle is the cycle in reverse.
      size_t i = *cycle.rbegin();

      // Copy the first "row"
      for (size_t j = jFirst; j <= jLast; ++j)
      {
        tmp[j] = matrix[idx(i, j, n)];
      }

      size_t iNext = 0;
      auto last = std::prev(cycle.rend());
      for (auto it = cycle.rbegin(); it != last; ++it)
      {
        i = *it;
        iNext = *std::next(it);
        for (size_t j = jFirst; j <= jLast; ++j)
        {
          matrix[idx(i, j, n)] = matrix[idx(iNext, j, n)];
        }
      }

      i = *cycle.rbegin();
      iNext = *last;
      for (size_t j = jFirst; j <= jLast; ++j)
      {
        matrix[idx(iNext, j, n)] = tmp[j];
      }
    }

  }

  namespace detail
  {
    // Ugly hack to keep flows alive across function boundaries
    struct ReversePermutationFlows
    {
      tf::Taskflow flow;

      tf::Taskflow rowFlow;
      tf::Taskflow colFlow;
    };
  }
  
  /// Important! cycles must stay alive for the duration of the flow
  template <typename Tv>
  inline detail::ReversePermutationFlows construct_reverse_permute_in_place_flow(Tv* matrix, size_t n, const std::vector<std::vector<size_t>>& cycles, size_t blockSz = 128ul)
  {
    detail::RowIndexer rows;
    detail::ColumnIndexer cols;

    //auto cycles = std::make_shared<std::vector<std::vector<size_t>>>(std::move(get_cycles(permutation)));
    auto divs = subdivide(blockSz, n);

    detail::ReversePermutationFlows flows;

    flows.rowFlow.for_each(cycles.begin(), cycles.end(), [matrix](const std::vector<size_t>& cycle){
      std::cout << "Cycle of len " << cycle.size() << std::endl;
      
    });
    flows.flow.composed_of(flows.rowFlow);
    
#if 0
    for (auto const& div : divs)
    {
      flows.rowFlow.for_each(cycles.begin(), cycles.end(), [matrix, n, rows, div](const std::vector<size_t>& cycle) {
        std::cout << "Unpermute ROW cycle of length " << cycle.size() << std::endl; 
        detail::apply_matrix_reverse_cycle(matrix, n, div.first, div.second, cycle, rows);
        std::cout << "Done unpermute ROW cycle of length " << cycle.size() << std::endl;
        });
      
    }
    flows.flow.composed_of(flows.rowFlow);
    #endif
    #if 0
    for (auto const& div : divs)
    {
      flows.colFlow.for_each(cycles.begin(), cycles.end(), [matrix, n, cols, div](const std::vector<size_t>& cycle) {
        std::cout << "Unpermute COL cycle of length " << cycle.size() << std::endl; 
        detail::apply_matrix_reverse_cycle(matrix, n, div.first, div.second, cycle, cols);
        std::cout << "Done unpermute COL cycle of length " << cycle.size() << std::endl;
        });
    }
    
    auto rowTask = flows.flow.composed_of(flows.rowFlow);
    auto colTask = flows.flow.composed_of(flows.colFlow);
    
    colTask.succeed(rowTask);
#endif
    return flows;
  }
  
  /// Given a 'matrix' that has been computed along a 'permutation', this function applies the inverse
  /// permutation of the rows and columns, respectively. Computations are done in-place, i.e., there is
  /// no new matrix allocated (although O(n) additional memory is required to complete the operation).
  /// 
  /// All cycles are expected to be nonempty.
  template <typename Tv>
  inline void reverse_permute_in_place(Tv* matrix, std::vector<size_t> permutation, Executor& exec = default_executor(), size_t blockSz = 128ul)
  {
    auto cycles = get_cycles(permutation);
    auto n = permutation.size();
    auto flows = construct_reverse_permute_in_place_flow(matrix, n, cycles, blockSz);
    exec.cpu()->run(std::move(flows.flow)).wait();
  }
}

#endif
