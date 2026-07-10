#ifndef STABLEBEAR_HOMOLOGICAL_KERNEL_H
#define STABLEBEAR_HOMOLOGICAL_KERNEL_H

#include "../task.hpp"
#include "barcode.hpp"
#include "distance_matrix.hpp"
#include "executor.hpp"
#include "taskflow/core/taskflow.hpp"
#include "tensor.hpp"
#include "walk.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
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

    template <typename T>
    struct Dendrogram
    {
      std::vector<size_t> parent; // size 2n−1; parent[root] == root
      std::vector<T> height;      // size 2n−1; 0 for leaves, merge weight for internal nodes
    };

    // Union-find root lookup with path compression.
    inline size_t uf_find_root(std::vector<size_t> &ufParent, size_t x)
    {
      auto root = x;
      while (ufParent[root] != root)
      {
        root = ufParent[root];
      }
      while (ufParent[x] != root) // path compression
      {
        auto next = ufParent[x];
        ufParent[x] = root;
        x = next;
      }
      return root;
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

    // this will use the MST constructed above to construct data in the form of a dendrogram. This will make it easy to
    // perform cross filtration later for the kernel.
    //
    // merges must be sorted ascending by mergeDist (as produced by mst_merge_order) and
    // hold exactly n-1 entries. Nodes 0..n-1 are leaves; merge i mints internal node n+i.
    template <typename T>
    void build_dendrogram(const std::vector<MergeEdge<T>> &merges, size_t n, Dendrogram<T> &tree)
    {
      if (n == 0)
      {
        tree.parent.clear();
        tree.height.clear();
        return;
      }

      auto numNodes = (2 * n) - 1;
      tree.parent.resize(numNodes);
      tree.height.assign(numNodes, T{0});
      for (size_t v = 0; v < numNodes; ++v)
      {
        tree.parent[v] = v; // every node is its own root until a later merge links it
      }

      // Scratch union-find over points. Path compression rewires parent links, so it
      // must not run on tree.parent itself: those links are the output.
      std::vector<size_t> ufParent(n);
      std::vector<size_t> clusterNode(n); // union-find representative -> current cluster top node
      for (size_t p = 0; p < n; ++p)
      {
        ufParent[p] = p;
        clusterNode[p] = p;
      }

      for (size_t i = 0; i < merges.size(); ++i)
      {
        auto ra = uf_find_root(ufParent, merges[i].a);
        auto rb = uf_find_root(ufParent, merges[i].b);

        auto v = n + i;
        tree.parent[clusterNode[ra]] = v;
        tree.parent[clusterNode[rb]] = v;
        tree.height[v] = merges[i].mergeDist;

        ufParent[rb] = ra;
        clusterNode[ra] = v;
      }
    }

    // Performs cross filtration on the dendrogram to calculate the kernel barcode.
    //
    // Replays the d' merges (births w_i) and computes each death
    //   v_i = min over cross pairs (x in A_i, y in B_i) of the d-tree LCA height,
    // i.e. the contracted subdominant ultrametric of d. Uses two facts:
    //  1. with leaves in DFS order, the LCA height of two leaves is the maximum of
    //     the gap array (LCA heights of consecutive leaves) over the spanned range;
    //  2. the minimizing cross pair is adjacent in the merged position order, so a
    //     merge only has to check each smaller-side position against its neighbours
    //     in the larger side's ordered set (small-to-large).
    // O(n log^2 n) time, O(n log n) memory. Throws if some death lands below its
    // birth, which means d does not dominate d'.
    template <typename T>
    void cross_filtration(
        const std::vector<MergeEdge<T>> &primeMerges, const Dendrogram<T> &dTree, size_t n,
        std::vector<PersistencePair<T>> &bars)
    {
      bars.clear();
      if (n <= 1)
      {
        return;
      }
      bars.reserve(n - 1);

      const auto numNodes = (2 * n) - 1;

      // Child lists from the parent array. The tree is binary: two slots per node.
      std::vector<std::array<size_t, 2>> children(numNodes);
      std::vector<unsigned char> childCount(numNodes, 0);
      for (size_t v = 0; v < numNodes; ++v)
      {
        auto p = dTree.parent[v];
        if (p != v)
        {
          children[p][childCount[p]++] = v;
        }
      }

      // DFS from the root: pos[leaf] = position in leaf order, and gap[k] = height of
      // the divergence node between consecutive leaves k and k+1 (their LCA).
      std::vector<size_t> pos(n);
      std::vector<T> gap(n - 1);
      {
        std::vector<std::pair<size_t, size_t>> stack; // (node, next child slot)
        stack.reserve(numNodes);
        stack.emplace_back(numNodes - 1, 0); // last minted node is the root
        size_t nextPos = 0;
        T pendingGap{0};
        while (!stack.empty())
        {
          auto v = stack.back().first;
          if (childCount[v] == 0) // leaf
          {
            if (nextPos > 0)
            {
              gap[nextPos - 1] = pendingGap;
            }
            pos[v] = nextPos++;
            stack.pop_back();
          }
          else if (stack.back().second < childCount[v])
          {
            auto slot = stack.back().second++;
            if (slot > 0)
            {
              // Descending into the second subtree of v: v is the divergence node
              // between the previous leaf and the next one to be emitted.
              pendingGap = dTree.height[v];
            }
            stack.emplace_back(children[v][slot], 0);
          }
          else
          {
            stack.pop_back();
          }
        }
      }

      // Sparse table over the gap array: table[j][k] = max of gap[k .. k + 2^j).
      const auto numGaps = n - 1;
      const auto levels = static_cast<size_t>(std::bit_width(numGaps));
      std::vector<std::vector<T>> table(levels);
      table[0].assign(gap.begin(), gap.end());
      for (size_t j = 1; j < levels; ++j)
      {
        const auto span = size_t{1} << j;
        table[j].resize(numGaps - span + 1);
        for (size_t k = 0; k + span <= numGaps; ++k)
        {
          table[j][k] = std::max(table[j - 1][k], table[j - 1][k + (span / 2)]);
        }
      }

      // LCA height of the leaves at positions lo < hi = max of gap[lo .. hi).
      auto lcaHeight = [&table](size_t lo, size_t hi) {
        const auto j = static_cast<size_t>(std::bit_width(hi - lo)) - 1;
        return std::max(table[j][lo], table[j][hi - (size_t{1} << j)]);
      };

      // Replay the d' merges, keeping per component the ordered set of its leaf positions.
      std::vector<size_t> ufParent(n);
      std::vector<std::set<size_t>> positions(n);
      for (size_t p = 0; p < n; ++p)
      {
        ufParent[p] = p;
        positions[p].insert(pos[p]);
      }

      for (auto const &merge : primeMerges)
      {
        auto ra = uf_find_root(ufParent, merge.a);
        auto rb = uf_find_root(ufParent, merge.b);
        if (positions[ra].size() < positions[rb].size())
        {
          std::swap(ra, rb); // ra keeps the larger position set
        }
        auto &large = positions[ra];
        auto &small = positions[rb];

        auto death = std::numeric_limits<T>::infinity();
        for (auto p : small)
        {
          auto it = large.lower_bound(p);
          if (it != large.end())
          {
            death = std::min(death, lcaHeight(p, *it));
          }
          if (it != large.begin())
          {
            death = std::min(death, lcaHeight(*std::prev(it), p));
          }
        }

        if (death < merge.mergeDist)
        {
          throw std::runtime_error("homological kernel: d and d' are not lipshitz (death below birth)");
        }
        bars.emplace_back(merge.mergeDist, death);

        large.insert(small.begin(), small.end());
        small.clear();
        ufParent[rb] = ra;
      }
    }

    // Calls the 3 steps above to perform the algorithm for the homological kernel.
    // d must be strictly larger than d' (d' <= d pointwise); cross_filtration throws otherwise.
    // For n <= 1 the kernel barcode is empty.
    template <typename DistT, typename T>
    void homological_kernel_single_impl(const DistT &dDist, const DistT &dPrimeDist, size_t n, Barcode<T> &ret)
    {
      std::vector<MergeEdge<T>> primeMerges; // d' merge order: the births
      std::vector<MergeEdge<T>> dMerges;
      mst_merge_order(dPrimeDist, n, primeMerges);
      mst_merge_order(dDist, n, dMerges);

      Dendrogram<T> tree; // only d needs a dendrogram: it answers the death queries
      build_dendrogram(dMerges, n, tree);

      std::vector<PersistencePair<T>> bars;
      cross_filtration(primeMerges, tree, n, bars);
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
        throw std::runtime_error("homological kernel inputs must have the same shape");
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

      // validation here: rank()==2 for both, pc.shape(0)==pcPrime.shape(0)
      // (mirror the throw at compute_persistence.hpp:107-110)
      if (pc.shape() != pcPrime.shape())
      {
        throw std::runtime_error("homological kernel inputs must have the same shape");
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
    // TODO Test results against handcalculated examples and old paper code

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
        throw std::runtime_error("homological kernel inputs must have the same shape");

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
