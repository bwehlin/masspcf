#ifndef MPCF_ALGORITHM_PERMUTE_IN_PLACE_H
#define MPCF_ALGORITHM_PERMUTE_IN_PLACE_H

#include <vector>

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

  template <typename Tv>
  inline void permute_in_place(Tv* matrix, const std::vector<size_t>& permutation)
  {
    auto sz = permutation.size();

  }
}

#endif
