#ifndef MPCF_ALGORITHM_PERMUTE_IN_PLACE_H
#define MPCF_ALGORITHM_PERMUTE_IN_PLACE_H

#include <vector>

namespace mpcf
{
  /// Returns a cycle decomposition of the given permutation. It is assumed that
  /// if 'permutation' has size n, then all values 0,1,...,n-1 occur in 'permutation'
  std::vector<std::vector<size_t>> get_cycles(std::vector<size_t> permutation)
  {
    auto seen = permutation.size(); // Sentinel value
    std::vector<std::vector<size_t>> cycles;

    std::vector<size_t> cycle;
    for (size_t i = 0ul; i < permutation.size(); ++i)
    {
      if (permutation[i] == seen) 
      {
        continue;
      }

      cycle.clear();
      cycle.emplace_back(i);
      auto n = permutation[i];
      while (n != seen && n != i)
      {
        cycle.emplace_back(n);

        auto nextn = permutation[n];
        permutation[n] = seen;

        n = nextn;
      }

      if (cycle.size() > 1)
      {
        cycles.emplace_back();
        cycles.back().swap(cycle);
      }
    }

    return cycles;
  }

  template <typename Tv>
  void permute_in_place(Tv* matrix, const std::vector<size_t>& permutation)
  {
    auto sz = permutation.size();

  }
}

#endif
