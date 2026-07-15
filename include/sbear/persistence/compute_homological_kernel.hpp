#ifndef STABLEBEAR_HOMOLOGICAL_KERNEL_H
#define STABLEBEAR_HOMOLOGICAL_KERNEL_H

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
    /// Euclidean distance functor over a point cloud: answers d(i, j) on demand
    /// without materializing a distance matrix. Exposes the same call interface
    /// as DistanceMatrix<T> (operator()(i, j) and size()), so the kernel
    /// algorithms can be templated over either.
    template <typename T>
    class EuclideanDistance
    {
    public:
      explicit EuclideanDistance(const PointCloud<T> &points)
          : m_data(points.data()), m_pointStride(points.stride(0)), m_coordStride(points.stride(1)),
            m_size(points.shape(0)), m_dim(points.shape(1))
      {
      }

      T operator()(size_t i, size_t j) const
      {
        const T *p = m_data + static_cast<ptrdiff_t>(i) * m_pointStride;
        const T *q = m_data + static_cast<ptrdiff_t>(j) * m_pointStride;
        T sumSq{0};
        for (auto k = ptrdiff_t{0}; k < static_cast<ptrdiff_t>(m_dim); ++k)
        {
          auto diff = p[k * m_coordStride] - q[k * m_coordStride];
          sumSq += diff * diff;
        }
        return std::sqrt(sumSq);
      }

      [[nodiscard]] size_t size() const noexcept
      {
        return m_size;
      }

    private:
      const T *m_data;
      ptrdiff_t m_pointStride;
      ptrdiff_t m_coordStride;
      size_t m_size;
      size_t m_dim;
    };

    template <typename T>
    struct MergeEdge
    {
      size_t a; // endpoints of the MST edge = representatives of the two merging components
      size_t b;
      T mergeDist;
    };

    // Union-find root lookup with path halving: every node on the walk is
    // pointed at its grandparent, halving the path in one traversal. Same
    // near-constant amortized bound as two-pass compression, but one pass.
    inline size_t uf_find_root(std::vector<size_t> &ufParent, size_t x)
    {
      while (ufParent[x] != x)
      {
        ufParent[x] = ufParent[ufParent[x]];
        x = ufParent[x];
      }
      return x;
    }

    // Distance type is in practice a distance matrix. The euclideanDistance above calculates the distance for a
    // pointcloud as well so we dont even need to construct a full distance matrix when working with pointclouds.
    //
    // Fills merges with the n-1 single-linkage merges of the metric, sorted ascending
    // by merge distance. Array-based Prim: O(n^2) distance queries, O(n) memory.
    template <typename DistT, typename T>
    void mst_merge_order(const DistT &dist, size_t n, std::vector<MergeEdge<T>> &merges)
    {
      merges.clear();
      if (n <= 1)
        return;
      merges.reserve(n - 1);

      std::vector<T> minDist(n, std::numeric_limits<T>::infinity());
      std::vector<size_t> parent(n, 0);
      std::vector<char> visited(n, 0);

      minDist[0] = T{0};

      for (size_t iter = 0; iter < n; ++iter)
      {
        // Pick the unvisited node closest to the tree.
        size_t u = 0;
        bool found = false;
        for (size_t j = 0; j < n; ++j)
        {
          if (!visited[j] && (!found || minDist[j] < minDist[u]))
          {
            u = j;
            found = true;
          }
        }

        visited[u] = 1;
        if (iter > 0)
        {
          merges.push_back({parent[u], u, minDist[u]});
        }

        for (size_t j = 0; j < n; ++j)
        {
          if (!visited[j])
          {
            auto d = dist(u, j);
            if (d < minDist[j])
            {
              minDist[j] = d;
              parent[j] = u;
            }
          }
        }
      }

      // Lambda to order the merges by size. Neccecary for later steps
      std::sort(merges.begin(), merges.end(), [](const MergeEdge<T> &x, const MergeEdge<T> &y) {
        return x.mergeDist < y.mergeDist;
      });
    }

    // Performs the cross filtration: replays the d' merges (the births w_i) and
    // computes each death v_i against d.
    //
    // The death of merge i is the scale at which [a_i] and [b_i] connect in the
    // quotient of (X, d) by the i-1 earlier contractions — the subdominant ultra
    // pseudo-metric of the quotient space. Contracting two d-far points (which
    // every nonempty bar does) creates shortcuts THROUGH the contracted
    // component that lower the connection scale of other pairs, so the death is
    // a global property of the contracted space: it cannot be read off the
    // static d-dendrogram (the min-cross-pair LCA there equals the Lance-
    // Williams min-rule, which overestimates deaths), and no bounded-radius
    // search around the merging pair can answer it either — the connecting
    // chain may hop through arbitrarily many other contracted components.
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
        // runs at least once and exits with k <= n-1, having set death.
        //
        // The connectivity check is find-free: rootA/rootB are cached, and a
        // cached root can only stop being one when the union writes over its
        // slot (rv below) — so patching on that case keeps both caches exact.
        auto rootA = uf_find_root(scratchParent, merge.a);
        auto rootB = uf_find_root(scratchParent, merge.b);
        auto death = std::numeric_limits<T>::quiet_NaN();
        size_t k = 0;
        while (rootA != rootB)
        {
          auto const &edge = dMerges[k++];
          // Unconditional union: an edge whose endpoints are already connected
          // (through the seed or earlier edges) is a harmless self-assignment.
          auto ru = uf_find_root(scratchParent, edge.a);
          auto rv = uf_find_root(scratchParent, edge.b);
          scratchParent[rv] = ru;
          if (rv == rootA)
          {
            rootA = ru;
          }
          else if (rv == rootB)
          {
            rootB = ru;
          }
          death = edge.mergeDist;
        }

        // Births (from d') and deaths (from d) can reach mathematically equal
        // values through different arithmetic. The gap is not always a few ULPs
        // of the distances: inputs materialized at large coordinate magnitude
        // (e.g. projected point clouds) carry absolute error ~eps * |coordinate|,
        // which dwarfs ULP-of-distance whenever coordinates dwarf pair
        // distances. sqrt(eps) relative splits the significand between signal
        // and guard: it forgives any plausible roundoff, while genuine
        // domination failures -- which violate at data scale -- still throw.
        const auto tol =
            std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(std::abs(merge.mergeDist), std::abs(death));
        if (death < merge.mergeDist - tol)
        {
          throw std::runtime_error("homological kernel: d does not dominate d' (death below birth)");
        }
        bars.emplace_back(merge.mergeDist, std::max(death, merge.mergeDist));

        // Only now does the contraction enter the seed: merge i must see the
        // quotient by merges 1..i-1, not by itself or later ones.
        seedParent[uf_find_root(seedParent, merge.b)] = uf_find_root(seedParent, merge.a);
      }
    }

    // Calls the 3 steps above to perform the algorithm for the homological kernel.
    // d must be strictly larger than d' (d' <= d pointwise); cross_filtration throws otherwise.
    // For n <= 1 the kernel barcode is empty.
    template <typename DistT, typename T>
    void homological_kernel_single_impl(const DistT &dDist, const DistT &dPrimeDist, size_t n, Barcode<T> &ret)
    {
      std::vector<MergeEdge<T>> primeMerges; // d' merge order: the births
      std::vector<MergeEdge<T>> dMerges;     // d-MST edges: the sweep timeline that answers the deaths
      mst_merge_order(dPrimeDist, n, primeMerges);
      mst_merge_order(dDist, n, dMerges);

      std::vector<PersistencePair<T>> bars;
      cross_filtration(primeMerges, dMerges, n, bars);
      ret = std::move(bars);
    }

    template <typename T>
    void homological_kernel_distmat_single_impl(
        const Tensor<DistanceMatrix<T>> &distmat, const Tensor<DistanceMatrix<T>> &distmatPrime,
        Tensor<Barcode<T>> &ret_barcodes, const std::vector<size_t> &index)
    {
      auto const &dm = distmat(index);           // the Distmat<T> for this instance
      auto const &dmPrime = distmatPrime(index); // its aligned d′ counterpart
      if (dm.size() != dmPrime.size())
      {
        throw std::runtime_error(
            "homological kernel: distance matrices at index " + index_to_string(index) + " have mismatched sizes (" +
            std::to_string(dm.size()) + " and " + std::to_string(dmPrime.size()) + ")");
      }

      detail::homological_kernel_single_impl(dm, dmPrime, dm.size(), ret_barcodes(index));
    }

    template <typename T>
    void homological_kernel_pcloud_single_impl(
        const Tensor<PointCloud<T>> &pclouds, const Tensor<PointCloud<T>> &pcloudsPrime,
        Tensor<Barcode<T>> &ret_barcodes, const std::vector<size_t> &index)
    {
      auto const &pc = pclouds(index);           // the PointCloud<T> for this instance
      auto const &pcPrime = pcloudsPrime(index); // its aligned d′ counterpart

      if (pc.shape() != pcPrime.shape())
      {
        throw std::runtime_error(
            "homological kernel: point clouds at index " + index_to_string(index) + " have mismatched shapes " +
            shape_to_string(pc.shape()) + " and " + shape_to_string(pcPrime.shape()));
      }

      // Degenerate clouds (no points or no coordinates) have an empty kernel.
      if (pc.rank() == 0 || std::any_of(pc.shape().begin(), pc.shape().end(), [](size_t v) { return v == 0; }))
      {
        return;
      }

      if (pc.rank() != 2)
      {
        throw std::runtime_error(
            "homological kernel: point cloud at index " + index_to_string(index) + " has unexpected shape " +
            shape_to_string(pc.shape()) + " (should be (m, n))");
      }

      detail::EuclideanDistance<T> dDist(pc); // captures pointer+strides, computes d on demand
      detail::EuclideanDistance<T> dPrimeDist(pcPrime);

      detail::homological_kernel_single_impl(dDist, dPrimeDist, dDist.size(), ret_barcodes(index));
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
      auto shape = m_input.shape();
      m_ret = Tensor<Barcode<T>>(shape);

      if (m_input.shape() != m_inputPrime.shape())
        throw std::runtime_error(
            "homological kernel: input tensors must have the same shape (got " + shape_to_string(m_input.shape()) +
            " and " + shape_to_string(m_inputPrime.shape()) + ")");

      next_step(
          m_input.size(), "Computing 0th homological kernel",
          std::is_same_v<ElemT, PointCloud<T>> ? "pointcloud" : "distance matrix");
      return sb::parallel_walk_async(
          m_input,
          [this](const std::vector<size_t> &index) {
            if (stop_requested())
              return;

            if constexpr (std::is_same_v<ElemT, PointCloud<T>>)
              detail::homological_kernel_pcloud_single_impl(m_input, m_inputPrime, m_ret, index);
            else
              detail::homological_kernel_distmat_single_impl(m_input, m_inputPrime, m_ret, index);
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
