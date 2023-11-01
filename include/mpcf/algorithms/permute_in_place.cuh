#ifndef MPCF_ALGORITHM_PERMUTE_IN_PLACE_H
#define MPCF_ALGORITHM_PERMUTE_IN_PLACE_H

#include <vector>

#include "../executor.h"

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
    inline void apply_matrix_reverse_cycle(Tv* matrix, size_t n, std::vector<Tv>& tmp, const std::vector<size_t>& cycle, Indexer idx, Executor& exec) 
    {
      // This code is written as if we are flipping rows of the matrix, but the indexer allows for flipping columns instead.

      // The inverse of a cycle is the cycle in reverse.
      size_t i = *cycle.rbegin();

      // Copy the first "row"
      for (size_t j = 0ul; j < n; ++j)
      {
        tmp[j] = matrix[idx(i, j, n)];
      }

      size_t iNext = 0;
      auto last = std::prev(cycle.rend());
      for (auto it = cycle.rbegin(); it != last; ++it)
      {
        i = *it;
        iNext = *std::next(it);
        for (size_t j = 0ul; j < n; ++j)
        {
          matrix[idx(i, j, n)] = matrix[idx(iNext, j, n)];
        }
      }

      i = *cycle.rbegin();
      iNext = *last;
      for (size_t j = 0ul; j < n; ++j)
      {
        matrix[idx(iNext, j, n)] = tmp[j];
      }
    }

  }

  /// Given a 'matrix' that has been computed along a 'permutation', this function applies the inverse
  /// permutation of the rows and columns, respectively. Computations are done in-place, i.e., there is
  /// no new matrix allocated (although O(n) additional memory is required to complete the operation).
  /// 
  /// All cycles are expected to be nonempty.
  template <typename Tv>
  inline void reverse_permute_in_place(Tv* matrix, std::vector<size_t> permutation, Executor& exec = default_executor())
  {
    auto n = permutation.size();

    detail::RowIndexer rows;
    detail::ColumnIndexer cols;

    auto cycles = get_cycles(permutation);

    std::vector<Tv> tmp(n);
    for (auto const& cycle : cycles)
    {
      detail::apply_matrix_reverse_cycle(matrix, n, tmp, cycle, rows, exec);
    }

    for (auto const& cycle : cycles)
    {
      detail::apply_matrix_reverse_cycle(matrix, n, tmp, cycle, cols, exec);
    }

  }
}

#endif
