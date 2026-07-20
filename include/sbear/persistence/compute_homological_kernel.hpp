#ifndef STABLEBEAR_HOMOLOGICAL_KERNEL_H
#define STABLEBEAR_HOMOLOGICAL_KERNEL_H

#include "../algorithms/minimum_spanning_tree.hpp"
#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../task.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"
#include "barcode.hpp"

#include "taskflow/core/taskflow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sb::ph
{
  namespace detail
  {
    // Union-find root lookup with path halving.
    inline size_t uf_find_root(std::vector<size_t> &ufParent, size_t x)
    {
      while (ufParent[x] != x)
      {
        ufParent[x] = ufParent[ufParent[x]];
        x = ufParent[x];
      }
      return x;
    }

    // Performs the cross filtration: replays the d' merges (the births w_i) and
    // computes each death v_i against d.
    //
    // The death of merge i is the scale at which [a_i] and [b_i] connect in the
    // quotient of (X, d) by the i-1 earlier contractions — the subdominant ultra
    // pseudo-metric of the quotient space. A contraction can create a shortcut
    // through the contracted component that lowers the connection scale of
    // other pairs, so each death depends on the whole set of earlier
    // contractions, not just the merging pair.
    //
    // Connectivity at scale t of (X, d) equals connectivity of the d-MST
    // restricted to edges <= t, and contractions only add identifications on
    // top. So each death is answered by a Kruskal sweep: seed a union-find with
    // the current d'-components, add d-MST edges in ascending order, and stop
    // as soon as a_i ~ b_i; the weight of the connecting edge is the death.
    //
    // The sweep for merge i stops after rank(v_i) edges, and the deaths are a
    // permutation of the d-MST edge ranks, so all sweeps together perform
    // exactly n(n-1)/2 bounded union-find operations plus an O(n) seed copy per
    // merge: O(n^2 alpha(n)) time, O(n) memory. Throws if some death lands
    // below its birth (beyond roundoff), which means d does not dominate d'.
    template <typename T>
    void cross_filtration(
        const std::vector<MergeEdge<T>> &primeMerges, const std::vector<MergeEdge<T>> &dMerges, size_t n,
        std::vector<PersistencePair<T>> &bars)
    {
      bars.clear();
      if (n <= 1)
      {
        return;
      }
      bars.reserve(n - 1);

      // seedParent holds the components induced by the contractions performed so
      // far (= the current d'-components); scratchParent is the working copy each
      // sweep runs on. The copy assignment reuses capacity, so the steady state
      // allocates nothing.
      std::vector<size_t> seedParent(n);
      for (size_t p = 0; p < n; ++p)
      {
        seedParent[p] = p;
      }
      std::vector<size_t> scratchParent;

      for (auto const &merge : primeMerges)
      {
        scratchParent = seedParent;

        // The seed alone can never connect merge.a to merge.b (its unions stay
        // inside d'-components, and a and b are in different ones by definition
        // of this merge), and the full d-MST is spanning — so the loop always
        // runs at least once and exits with edgeIdx <= n-1, having set death.
        //
        // The connectivity check is find-free: rootA/rootB are cached, and a
        // cached root can only stop being one when the union writes over its
        // slot (absorbedRoot below) — so patching on that case keeps both
        // caches exact.
        auto rootA = uf_find_root(scratchParent, merge.a);
        auto rootB = uf_find_root(scratchParent, merge.b);
        auto death = std::numeric_limits<T>::quiet_NaN();
        size_t edgeIdx = 0;
        while (rootA != rootB)
        {
          auto const &edge = dMerges[edgeIdx++];
          // Unconditional union: an edge whose endpoints are already connected
          // (through the seed or earlier edges) is a harmless self-assignment.
          auto survivingRoot = uf_find_root(scratchParent, edge.a);
          auto absorbedRoot = uf_find_root(scratchParent, edge.b);
          scratchParent[absorbedRoot] = survivingRoot;
          if (absorbedRoot == rootA)
          {
            rootA = survivingRoot;
          }
          else if (absorbedRoot == rootB)
          {
            rootB = survivingRoot;
          }
          death = edge.mergeDist;
        }

        // Births (from d') and deaths (from d) can reach mathematically equal
        // values through different arithmetic, and inputs materialized at large
        // coordinate magnitude (e.g. projected point clouds) carry absolute
        // error ~eps * |coordinate|, which can dwarf a few ULPs of the
        // distances. sqrt(eps) relative forgives any plausible roundoff, while
        // genuine domination failures — which violate at data scale — still
        // throw.
        const auto tolerance =
            std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(std::abs(merge.mergeDist), std::abs(death));
        if (death < merge.mergeDist - tolerance)
        {
          throw std::runtime_error("homological kernel: d does not dominate d' (death below birth)");
        }
        bars.emplace_back(merge.mergeDist, std::max(death, merge.mergeDist));

        // Only now does the contraction enter the seed: merge i must see the
        // quotient by merges 1..i-1, not by itself or later ones.
        seedParent[uf_find_root(seedParent, merge.b)] = uf_find_root(seedParent, merge.a);
      }
    }

    // Full kernel computation for one instance: d' merge order (the births),
    // d-MST (the death timeline), cross filtration. Requires d' <= d pointwise;
    // cross_filtration throws otherwise. For n <= 1 the kernel barcode is empty.
    // mergeDistTransform maps each stored merge distance from the scale the
    // DistT functor computes in to the scale of the output bars (sqrt for the
    // squared-Euclidean oracle, identity for oracles already in bar scale); it
    // must be monotone so the merge order is unchanged.
    template <typename DistT, typename T, typename PostF = std::identity>
    void homological_kernel_single_impl(
        const DistT &dDist, const DistT &dPrimeDist, size_t n, Barcode<T> &ret, PostF mergeDistTransform = {})
    {
      std::vector<MergeEdge<T>> primeMerges; // d' merge order: the births
      std::vector<MergeEdge<T>> dMerges;     // d-MST edges: the sweep timeline that answers the deaths
      mst_merge_order(dPrimeDist, n, primeMerges);
      mst_merge_order(dDist, n, dMerges);
      for (auto &m : primeMerges)
      {
        m.mergeDist = mergeDistTransform(m.mergeDist);
      }
      for (auto &m : dMerges)
      {
        m.mergeDist = mergeDistTransform(m.mergeDist);
      }

      std::vector<PersistencePair<T>> bars;
      cross_filtration(primeMerges, dMerges, n, bars);
      ret = std::move(bars);
    }

    template <typename T>
    void homological_kernel_distmat_single_impl(
        const Tensor<DistanceMatrix<T>> &distmat, const Tensor<DistanceMatrix<T>> &distmatPrime,
        Tensor<Barcode<T>> &retBarcodes, const std::vector<size_t> &index)
    {
      auto const &dm = distmat(index);           // the Distmat<T> for this instance
      auto const &dmPrime = distmatPrime(index); // its aligned d′ counterpart
      if (dm.size() != dmPrime.size())
      {
        throw std::runtime_error(
            "homological kernel: distance matrices at index " + index_to_string(index) + " have mismatched sizes (" +
            std::to_string(dm.size()) + " and " + std::to_string(dmPrime.size()) + ")");
      }

      detail::homological_kernel_single_impl(dm, dmPrime, dm.size(), retBarcodes(index));
    }

    template <typename T>
    void homological_kernel_pcloud_single_impl(
        const Tensor<PointCloud<T>> &pclouds, const Tensor<PointCloud<T>> &pcloudsPrime,
        Tensor<Barcode<T>> &retBarcodes, const std::vector<size_t> &index)
    {
      auto const &pc = pclouds(index);           // the PointCloud<T> for this instance
      auto const &pcPrime = pcloudsPrime(index); // its aligned d′ counterpart

      if (pc.shape() != pcPrime.shape())
      {
        throw std::runtime_error(
            "homological kernel: point clouds at index " + index_to_string(index) + " have mismatched shapes " +
            shape_to_string(pc.shape()) + " and " + shape_to_string(pcPrime.shape()));
      }

      if (pc.rank() != 2)
      {
        throw std::runtime_error(
            "homological kernel: point cloud at index " + index_to_string(index) + " has unexpected shape " +
            shape_to_string(pc.shape()) + " (should be (m, n))");
      }

      // Every rank-2 cloud flows through: 0 or 1 points give an empty barcode
      // via the n <= 1 early-outs, and a zero-dimensional cloud (n, 0) induces
      // the all-zero metric, yielding n-1 zero-length bars exactly like the
      // distance-matrix route does for the equivalent all-zero matrix.
      SquaredEuclideanDistance<T> dDist(pc); // captures pointer+strides, computes d^2 on demand
      SquaredEuclideanDistance<T> dPrimeDist(pcPrime);

      detail::homological_kernel_single_impl(
          dDist, dPrimeDist, dDist.size(), retBarcodes(index), [](T v) { return std::sqrt(v); });
    }

  } // namespace detail

  template <typename ElemT, typename T>
  class HomologicalKernelImpl : public StoppableTask<void>
  {
  public:
    HomologicalKernelImpl(const Tensor<ElemT> &input, const Tensor<ElemT> &inputPrime, Tensor<Barcode<T>> &ret)
        : m_input(input), m_inputPrime(inputPrime), m_ret(ret)
    {
    }

  private:
    tf::Future<void> run_async(Executor &exec) override
    {
      if (m_input.shape() != m_inputPrime.shape())
      {
        throw std::runtime_error(
            "homological kernel: input tensors must have the same shape (got " + shape_to_string(m_input.shape()) +
            " and " + shape_to_string(m_inputPrime.shape()) + ")");
      }

      // Validate before touching m_ret so a failed spawn leaves the caller's
      // out tensor intact.
      m_ret = Tensor<Barcode<T>>(m_input.shape());

      next_step(
          m_input.size(), "Computing 0th homological kernel",
          std::is_same_v<ElemT, PointCloud<T>> ? "pointcloud" : "distance matrix");
      return sb::parallel_walk_async(
          m_input,
          [this](const std::vector<size_t> &index) {
            if (stop_requested())
            {
              return;
            }

            if constexpr (std::is_same_v<ElemT, PointCloud<T>>)
            {
              detail::homological_kernel_pcloud_single_impl(m_input, m_inputPrime, m_ret, index);
            }
            else
            {
              detail::homological_kernel_distmat_single_impl(m_input, m_inputPrime, m_ret, index);
            }

            add_progress(1);
          },
          exec);
    }
    const Tensor<ElemT> &m_input;
    const Tensor<ElemT> &m_inputPrime;
    Tensor<Barcode<T>> &m_ret;
  };

} // namespace sb::ph

#endif
