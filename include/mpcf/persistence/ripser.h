/*

 Ripser: a lean C++ code for computation of Vietoris-Rips persistence barcodes

 MIT License

 Copyright (c) 2015–2021 Ulrich Bauer

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or
 upgrades to the features, functionality or performance of the source code
 ("Enhancements") to anyone; however, if you choose to make your Enhancements
 available either publicly, or directly to the author of this software, without
 imposing a separate written license agreement for such Enhancements, then you
 hereby grant the following license: a non-exclusive, royalty-free perpetual
 license to install, use, modify, prepare derivative works, incorporate into
 other computer software, distribute, and sublicense such enhancements or
 derivative works thereof, in binary and source code form.

*/

#pragma once

//#define USE_COEFFICIENTS

//#define INDICATE_PROGRESS
//#define PRINT_PERSISTENCE_PAIRS

//#define USE_ROBINHOOD_HASHMAP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>

#include "persistence_pair.h"

//#define INDICATE_PROGRESS

namespace rp
{

  #ifdef USE_ROBINHOOD_HASHMAP

  #include <robin_hood.h>

  template <class Key, class T, class H, class E>
  using hash_map = robin_hood::unordered_map<Key, T, H, E>;
  template <class Key> using hash = robin_hood::hash<Key>;

  #else

  template <class Key, class T, class H, class E> using hash_map = std::unordered_map<Key, T, H, E>;
  template <class Key> using hash = std::hash<Key>;

  #endif

  typedef double value_t;
  typedef int64_t index_t;
  typedef uint16_t coefficient_t;

  #ifdef INDICATE_PROGRESS
  static const std::chrono::milliseconds time_step(40);
  #endif

  static const std::string clear_line("\r\033[K");

  static const size_t num_coefficient_bits = 8;

  static const index_t max_simplex_index =
      (index_t(1) << (8 * sizeof(index_t) - 1 - num_coefficient_bits)) - 1;

  inline void check_overflow(index_t i) {
    if
  #ifdef USE_COEFFICIENTS
        (i > max_simplex_index)
  #else
        (i < 0)
  #endif
      throw std::overflow_error("simplex index " + std::to_string((uint64_t)i) +
                                " in filtration is larger than maximum index " +
                                std::to_string(max_simplex_index));
  }

  class binomial_coeff_table {
    std::vector<std::vector<index_t>> B;


  public:
    binomial_coeff_table(index_t n, index_t k) : B(k + 1, std::vector<index_t>(n + 1, 0)) {
      for (index_t i = 0; i <= n; ++i) {
        B[0][i] = 1;
        for (index_t j = 1; j < std::min(i, k + 1); ++j)
          B[j][i] = B[j - 1][i - 1] + B[j][i - 1];
        if (i <= k) B[i][i] = 1;
        check_overflow(B[std::min(i >> 1, k)][i]);
      }
    }

    index_t operator()(index_t n, index_t k) const {
      //std::cout << __PRETTY_FUNCTION__ << " n: " << n << " k: " << k << " B.size(): " << B.size() << std::endl;

      //std::cout << "k: " << (k < B.size()) << std::endl;
      //std::cout << "n: " << (n < B[k].size()) << std::endl;
      //assert(n < static_cast<index_t>(B.size()) && k < static_cast<index_t>(B[n].size()) && n >= k - 1);
      assert(k < static_cast<index_t>(B.size()) && n < static_cast<index_t>(B[k].size()));

      return B[k][n];
    }
  };

  inline bool is_prime(const coefficient_t n) {
    if (!(n & 1) || n < 2) return n == 2;
    for (coefficient_t p = 3; p <= n / p; p += 2)
      if (!(n % p)) return false;
    return true;
  }

  inline std::vector<coefficient_t> multiplicative_inverse_vector(const coefficient_t m) {
    std::vector<coefficient_t> inverse(m);
    inverse[1] = 1;
    // m = a * (m / a) + m % a
    // Multipying with inverse(a) * inverse(m % a):
    // 0 = inverse(m % a) * (m / a) + inverse(a)  (mod m)
    for (coefficient_t a = 2; a < m; ++a) inverse[a] = m - (inverse[m % a] * (m / a)) % m;
    return inverse;
  }

  #ifdef USE_COEFFICIENTS

  struct entry_t {
    index_t index : 8 * sizeof(index_t) - num_coefficient_bits;
    coefficient_t coefficient : num_coefficient_bits;
    entry_t(index_t _index, coefficient_t _coefficient)
        : index(_index), coefficient(_coefficient) {}
    entry_t(index_t _index) : index(_index), coefficient(0) {}
    entry_t() : index(0), coefficient(0) {}
  };

  static_assert(sizeof(entry_t) == sizeof(index_t), "size of entry_t is not the same as index_t");

  entry_t make_entry(index_t i, coefficient_t c) { return entry_t(i, c); }
  index_t get_index(const entry_t& e) { return e.index; }
  index_t get_coefficient(const entry_t& e) { return e.coefficient; }
  void set_coefficient(entry_t& e, const coefficient_t c) { e.coefficient = c; }

  std::ostream& operator<<(std::ostream& stream, const entry_t& e) {
    stream << get_index(e) << ":" << get_coefficient(e);
    return stream;
  }

  #else

  typedef index_t entry_t;
  inline index_t get_index(const entry_t& i) { return i; }
  inline index_t get_coefficient(const entry_t& /*i*/) { return 1; }
  inline entry_t make_entry(index_t _index, coefficient_t /*_value*/) { return entry_t(_index); }
  inline void set_coefficient(entry_t& /*e*/, const coefficient_t /*c*/) {}

  #endif

  inline const entry_t& get_entry(const entry_t& e) { return e; }

  typedef std::pair<value_t, index_t> diameter_index_t;
  inline value_t get_diameter(const diameter_index_t& i) { return i.first; }
  inline index_t get_index(const diameter_index_t& i) { return i.second; }

  typedef std::pair<index_t, value_t> index_diameter_t;
  inline index_t get_index(const index_diameter_t& i) { return i.first; }
  inline value_t get_diameter(const index_diameter_t& i) { return i.second; }

  struct diameter_entry_t : std::pair<value_t, entry_t> {
    using std::pair<value_t, entry_t>::pair;
    diameter_entry_t(value_t _diameter, index_t _index, coefficient_t _coefficient)
        : diameter_entry_t(_diameter, make_entry(_index, _coefficient)) {}
    diameter_entry_t(const diameter_index_t& _diameter_index, coefficient_t _coefficient)
        : diameter_entry_t(get_diameter(_diameter_index),
                           make_entry(get_index(_diameter_index), _coefficient)) {}
    diameter_entry_t(const diameter_index_t& _diameter_index)
        : diameter_entry_t(get_diameter(_diameter_index),
                           make_entry(get_index(_diameter_index), 0)) {}
    diameter_entry_t(const index_t& _index) : diameter_entry_t(0, _index, 0) {}
  };

  inline const entry_t& get_entry(const diameter_entry_t& p) { return p.second; }
  inline entry_t& get_entry(diameter_entry_t& p) { return p.second; }
  inline index_t get_index(const diameter_entry_t& p) { return get_index(get_entry(p)); }
  inline coefficient_t get_coefficient(const diameter_entry_t& p) {
    return get_coefficient(get_entry(p));
  }
  inline const value_t& get_diameter(const diameter_entry_t& p) { return p.first; }
  inline void set_coefficient(diameter_entry_t& p, const coefficient_t c) {
    set_coefficient(get_entry(p), c);
  }

  template <typename Entry> struct greater_diameter_or_smaller_index_comp {
    bool operator()(const Entry& a, const Entry& b) {
      return greater_diameter_or_smaller_index(a, b);
    }
  };

  template <typename Entry> bool greater_diameter_or_smaller_index(const Entry& a, const Entry& b) {
    return (get_diameter(a) > get_diameter(b)) ||
           ((get_diameter(a) == get_diameter(b)) && (get_index(a) < get_index(b)));
  }

  enum compressed_matrix_layout { LOWER_TRIANGULAR, UPPER_TRIANGULAR };

  template <compressed_matrix_layout Layout> struct compressed_distance_matrix {
    std::vector<value_t> distances;
    std::vector<value_t*> rows;

    compressed_distance_matrix(std::vector<value_t>&& _distances)
        : distances(std::move(_distances)), rows((1 + std::sqrt(1 + 8 * distances.size())) / 2) {
      assert(distances.size() == size() * (size() - 1) / 2);
      init_rows();
    }

    template <typename DistanceMatrix>
    compressed_distance_matrix(const DistanceMatrix& mat)
        : distances(mat.size() * (mat.size() - 1) / 2), rows(mat.size()) {
      init_rows();

      for (size_t i = 1; i < size(); ++i)
        for (size_t j = 0; j < i; ++j) rows[i][j] = mat(i, j);
    }

    value_t operator()(const index_t i, const index_t j) const;
    size_t size() const { return rows.size(); }
    void init_rows();
  };

  typedef compressed_distance_matrix<LOWER_TRIANGULAR> compressed_lower_distance_matrix;
  typedef compressed_distance_matrix<UPPER_TRIANGULAR> compressed_upper_distance_matrix;

  template <> inline void compressed_lower_distance_matrix::init_rows() {
    value_t* pointer = &distances[0];
    for (size_t i = 1; i < size(); ++i) {
      rows[i] = pointer;
      pointer += i;
    }
  }

  template <> inline void compressed_upper_distance_matrix::init_rows() {
    value_t* pointer = &distances[0] - 1;
    for (size_t i = 0; i < size() - 1; ++i) {
      rows[i] = pointer;
      pointer += size() - i - 2;
    }
  }

  template <>
  inline value_t compressed_lower_distance_matrix::operator()(const index_t i, const index_t j) const {
    return i == j ? 0 : i < j ? rows[j][i] : rows[i][j];
  }

  template <>
  inline value_t compressed_upper_distance_matrix::operator()(const index_t i, const index_t j) const {
    return i == j ? 0 : i > j ? rows[j][i] : rows[i][j];
  }

  struct sparse_distance_matrix {
    std::vector<std::vector<index_diameter_t>> neighbors;

    index_t num_edges;

    sparse_distance_matrix(std::vector<std::vector<index_diameter_t>>&& _neighbors,
                           index_t _num_edges)
        : neighbors(std::move(_neighbors)), num_edges(_num_edges) {}

    template <typename DistanceMatrix>
    sparse_distance_matrix(const DistanceMatrix& mat, const value_t threshold)
        : neighbors(mat.size()), num_edges(0) {

      for (size_t i = 0; i < size(); ++i)
        for (size_t j = 0; j < size(); ++j)
          if (i != j) {
            auto d = mat(i, j);
            if (d <= threshold) {
              ++num_edges;
              neighbors[i].push_back({j, d});
            }
          }
    }

    value_t operator()(const index_t i, const index_t j) const {
      auto neighbor =
          std::lower_bound(neighbors[i].begin(), neighbors[i].end(), index_diameter_t{j, 0});
      return (neighbor != neighbors[i].end() && get_index(*neighbor) == j)
                 ? get_diameter(*neighbor)
                 : std::numeric_limits<value_t>::infinity();
    }

    size_t size() const { return neighbors.size(); }
  };

  struct euclidean_distance_matrix {
    std::vector<std::vector<value_t>> points;

    euclidean_distance_matrix(std::vector<std::vector<value_t>>&& _points)
        : points(std::move(_points)) {
      for (auto p : points) { assert(p.size() == points.front().size()); }
    }

    value_t operator()(const index_t i, const index_t j) const {
      assert(i < static_cast<index_t>(points.size()));
      assert(j < static_cast<index_t>(points.size()));
      return std::sqrt(std::inner_product(
          points[i].begin(), points[i].end(), points[j].begin(), value_t(), std::plus<value_t>(),
          [](value_t u, value_t v) { return (u - v) * (u - v); }));
    }

    size_t size() const { return points.size(); }
  };

  class union_find {
    std::vector<index_t> parent;
    std::vector<uint8_t> rank;

  public:
    union_find(const index_t n) : parent(n), rank(n, 0) {
      for (index_t i = 0; i < n; ++i) parent[i] = i;
    }

    index_t find(index_t x) {
      index_t y = x, z;
      while ((z = parent[y]) != y) y = z;
      while ((z = parent[x]) != y) {
        parent[x] = y;
        x = z;
      }
      return z;
    }

    void link(index_t x, index_t y) {
      if ((x = find(x)) == (y = find(y))) return;
      if (rank[x] > rank[y])
        parent[y] = x;
      else {
        parent[x] = y;
        if (rank[x] == rank[y]) ++rank[y];
      }
    }
  };

  template <typename T> T begin(std::pair<T, T>& p) { return p.first; }
  template <typename T> T end(std::pair<T, T>& p) { return p.second; }

  template <typename ValueType> class compressed_sparse_matrix {
    std::vector<size_t> bounds;
    std::vector<ValueType> entries;

    typedef typename std::vector<ValueType>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

  public:
    size_t size() const { return bounds.size(); }

    iterator_pair subrange(const index_t index) {
      return {entries.begin() + (index == 0 ? 0 : bounds[index - 1]),
              entries.begin() + bounds[index]};
    }

    void append_column() { bounds.push_back(entries.size()); }

    void push_back(const ValueType e) {
      assert(0 < size());
      entries.push_back(e);
      ++bounds.back();
    }
  };

  template <class Predicate>
  index_t get_max(index_t top, const index_t bottom, const Predicate pred) {
    if (!pred(top)) {
      index_t count = top - bottom;
      while (count > 0) {
        index_t step = count >> 1, mid = top - step;
        if (!pred(mid)) {
          top = mid - 1;
          count -= step + 1;
        } else
          count = step;
      }
    }
    return top;
  }

  template <typename DistanceMatrix> class ripser;

  template <typename DistanceMatrix>
  class simplex_coboundary_enumerator {
    index_t idx_below, idx_above, j, k;
    std::vector<index_t> vertices;
    diameter_entry_t simplex;
    const coefficient_t modulus;
    const compressed_lower_distance_matrix& dist;
    const binomial_coeff_table& binomial_coeff;
    const ripser<DistanceMatrix>& parent;

  public:
    simplex_coboundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
                                  const ripser<DistanceMatrix>& _parent);
    simplex_coboundary_enumerator(const ripser<DistanceMatrix>& _parent);
    void set_simplex(const diameter_entry_t _simplex, const index_t _dim);
    bool has_next(bool all_cofacets = true);
    diameter_entry_t next();

  };

  template <typename DistanceMatrix> class ripser {
    const DistanceMatrix dist;
    const index_t n, dim_max;
    const value_t threshold;
    const float ratio;
    const coefficient_t modulus;
    const binomial_coeff_table binomial_coeff;
    const std::vector<coefficient_t> multiplicative_inverse;
    mutable std::vector<diameter_entry_t> cofacet_entries;
    mutable std::vector<index_t> vertices;

    std::vector<std::vector<mpcf::PersistencePair>> intervals;

    friend class simplex_coboundary_enumerator<DistanceMatrix>;

    class simplex_boundary_enumerator {
    private:
      index_t idx_below, idx_above, j, k;
      diameter_entry_t simplex;
      index_t dim;
      const coefficient_t modulus;
      const binomial_coeff_table& binomial_coeff;
      const ripser& parent;

    public:
      simplex_boundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
                                  const ripser& _parent)
          : idx_below(get_index(_simplex)), idx_above(0), j(_parent.n - 1), k(_dim),
            simplex(_simplex), modulus(_parent.modulus), binomial_coeff(_parent.binomial_coeff),
            parent(_parent) {}

      simplex_boundary_enumerator(const index_t _dim, const ripser& _parent)
          : simplex_boundary_enumerator(-1, _dim, _parent) {}

      void set_simplex(const diameter_entry_t _simplex, const index_t _dim) {
        idx_below = get_index(_simplex);
        idx_above = 0;
        j = parent.n - 1;
        k = _dim;
        simplex = _simplex;
        dim = _dim;
      }

      bool has_next() { return (k >= 0); }

      diameter_entry_t next() {
        j = parent.get_max_vertex(idx_below, k + 1, j);

        index_t face_index = idx_above - binomial_coeff(j, k + 1) + idx_below;

        value_t face_diameter = parent.compute_diameter(face_index, dim - 1);

        coefficient_t face_coefficient =
            (k & 1 ? -1 + modulus : 1) * get_coefficient(simplex) % modulus;

        idx_below -= binomial_coeff(j, k + 1);
        idx_above += binomial_coeff(j, k);

        --k;

        return diameter_entry_t(face_diameter, face_index, face_coefficient);
      }
    };



    simplex_boundary_enumerator facets;
    simplex_coboundary_enumerator<DistanceMatrix> cofacets, cofacets1, cofacets2;

    struct entry_hash {
      std::size_t operator()(const rp::entry_t& e) const { return std::hash<long>()(get_index(e)); }
    };

    struct equal_index {
      bool operator()(const rp::entry_t& e, const rp::entry_t& f) const {
        return get_index(e) == get_index(f);
      }
    };

    typedef hash_map<entry_t, size_t, entry_hash, equal_index> entry_hash_map;

  public:
    ripser(DistanceMatrix&& _dist, index_t _dim_max, value_t _threshold, float _ratio,
           coefficient_t _modulus)
        : dist(std::move(_dist)), n(dist.size()),
          dim_max(std::min(_dim_max, index_t(dist.size() - 2))), threshold(_threshold),
          ratio(_ratio), modulus(_modulus), binomial_coeff(n, dim_max + 2),
          multiplicative_inverse(multiplicative_inverse_vector(_modulus)),
          facets(0, *this),
          cofacets(*this),
          cofacets1(*this),
          cofacets2(*this)
    {
      intervals.resize(_dim_max + 1);
    }

    coefficient_t get_modulus() const
    {
      return modulus;
    }

    auto const & get_intervals(size_t dim)
    {
      return intervals[dim];
    }

    index_t get_max_vertex(const index_t idx, const index_t k, const index_t n) const {
      return get_max(n, k - 1, [&](index_t w) -> bool { return (binomial_coeff(w, k) <= idx); });
    }

    index_t get_edge_index(const index_t i, const index_t j) const {
      return binomial_coeff(i, 2) + j;
    }

    template <typename OutputIterator>
    OutputIterator get_simplex_vertices(index_t idx, const index_t dim, index_t n,
                                        OutputIterator out) const {
      --n;
      for (index_t k = dim + 1; k > 1; --k) {
        n = get_max_vertex(idx, k, n);
        *out++ = n;
        idx -= binomial_coeff(n, k);
      }
      *out = idx;
      return out;
    }

    value_t compute_diameter(const index_t index, const index_t dim) const {
      value_t diam = -std::numeric_limits<value_t>::infinity();

      vertices.resize(dim + 1);
      get_simplex_vertices(index, dim, dist.size(), vertices.rbegin());

      for (index_t i = 0; i <= dim; ++i)
        for (index_t j = 0; j < i; ++j) {
          diam = std::max(diam, dist(vertices[i], vertices[j]));
        }
      return diam;
    }



    diameter_entry_t get_zero_pivot_facet(const diameter_entry_t simplex, const index_t dim) {
      //static
      //simplex_boundary_enumerator facets(0, *this);
      facets.set_simplex(simplex, dim);
      while (facets.has_next()) {
        diameter_entry_t facet = facets.next();
        if (get_diameter(facet) == get_diameter(simplex)) return facet;
      }
      return diameter_entry_t(-1);
    }

    diameter_entry_t get_zero_pivot_cofacet(const diameter_entry_t simplex, const index_t dim) {
      //static
      //simplex_coboundary_enumerator cofacets(*this);
      cofacets.set_simplex(simplex, dim);
      while (cofacets.has_next()) {
        diameter_entry_t cofacet = cofacets.next();
        if (get_diameter(cofacet) == get_diameter(simplex)) return cofacet;
      }
      return diameter_entry_t(-1);
    }

    diameter_entry_t get_zero_apparent_facet(const diameter_entry_t simplex, const index_t dim) {
      diameter_entry_t facet = get_zero_pivot_facet(simplex, dim);
      return ((get_index(facet) != -1) &&
              (get_index(get_zero_pivot_cofacet(facet, dim - 1)) == get_index(simplex)))
                 ? facet
                 : diameter_entry_t(-1);
    }

    diameter_entry_t get_zero_apparent_cofacet(const diameter_entry_t simplex, const index_t dim) {
      diameter_entry_t cofacet = get_zero_pivot_cofacet(simplex, dim);
      return ((get_index(cofacet) != -1) &&
              (get_index(get_zero_pivot_facet(cofacet, dim + 1)) == get_index(simplex)))
                 ? cofacet
                 : diameter_entry_t(-1);
    }

    bool is_in_zero_apparent_pair(const diameter_entry_t simplex, const index_t dim) {
      return (get_index(get_zero_apparent_cofacet(simplex, dim)) != -1) ||
             (get_index(get_zero_apparent_facet(simplex, dim)) != -1);
    }

    void assemble_columns_to_reduce(std::vector<diameter_index_t>& simplices,
                                    std::vector<diameter_index_t>& columns_to_reduce,
                                    entry_hash_map& pivot_column_index, index_t dim) {

  #ifdef INDICATE_PROGRESS
      std::cerr << clear_line << "assembling columns" << std::flush;
      std::chrono::steady_clock::time_point next = std::chrono::steady_clock::now() + time_step;
  #endif

      columns_to_reduce.clear();
      std::vector<diameter_index_t> next_simplices;

      simplex_coboundary_enumerator cofacets(*this);

      for (diameter_index_t& simplex : simplices) {
        cofacets.set_simplex(diameter_entry_t(simplex, 1), dim - 1);

        while (cofacets.has_next(false)) {
  #ifdef INDICATE_PROGRESS
          if (std::chrono::steady_clock::now() > next) {
            std::cerr << clear_line << "assembling " << next_simplices.size()
                      << " columns (processing " << std::distance(&simplices[0], &simplex)
                      << "/" << simplices.size() << " simplices)" << std::flush;
            next = std::chrono::steady_clock::now() + time_step;
          }
  #endif
          auto cofacet = cofacets.next();
          if (get_diameter(cofacet) <= threshold) {
            if (dim < dim_max) next_simplices.push_back({get_diameter(cofacet), get_index(cofacet)});
            if (!is_in_zero_apparent_pair(cofacet, dim) &&
                (pivot_column_index.find(get_entry(cofacet)) == pivot_column_index.end()))
              columns_to_reduce.push_back({get_diameter(cofacet), get_index(cofacet)});
          }
        }
      }

      if (dim < dim_max) simplices.swap(next_simplices);

  #ifdef INDICATE_PROGRESS
      std::cerr << clear_line << "sorting " << columns_to_reduce.size() << " columns"
                << std::flush;
  #endif

      std::sort(columns_to_reduce.begin(), columns_to_reduce.end(),
                greater_diameter_or_smaller_index<diameter_index_t>);
  #ifdef INDICATE_PROGRESS
      std::cerr << clear_line << std::flush;
  #endif
    }

    void compute_dim_0_pairs(std::vector<diameter_index_t>& edges,
                             std::vector<diameter_index_t>& columns_to_reduce) {
  #ifdef PRINT_PERSISTENCE_PAIRS
      std::cout << "persistence intervals in dim 0:" << std::endl;
  #endif
      union_find dset(n);
      edges = get_edges();
      std::sort(edges.rbegin(), edges.rend(),
                greater_diameter_or_smaller_index<diameter_index_t>);
      std::vector<index_t> vertices_of_edge(2);
      value_t diameter = 0;
      for (auto e : edges) {
        get_simplex_vertices(get_index(e), 1, n, vertices_of_edge.rbegin());
        index_t u = dset.find(vertices_of_edge[0]), v = dset.find(vertices_of_edge[1]);

        if (u != v) {
          diameter = get_diameter(e);
          if (diameter != 0)
          {
  #ifdef PRINT_PERSISTENCE_PAIRS
            std::cout << " [0," << get_diameter(e) << ")" << std::endl;
  #endif
            intervals[0].emplace_back(0, diameter);
          }

          dset.link(u, v);
        } else if ((dim_max > 0) && (get_index(get_zero_apparent_cofacet(e, 1)) == -1))
          columns_to_reduce.push_back(e);
      }
      if (dim_max > 0) std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());

  #ifdef PRINT_PERSISTENCE_PAIRS
      for (index_t i = 0; i < n; ++i)
        if (dset.find(i) == i) std::cout << " [0, )" << std::endl;
  #endif
    }

    template <typename Column> diameter_entry_t pop_pivot(Column& column) {
      diameter_entry_t pivot(-1);
  #ifdef USE_COEFFICIENTS
      while (!column.empty()) {
        if (get_coefficient(pivot) == 0)
          pivot = column.top();
        else if (get_index(column.top()) != get_index(pivot))
          return pivot;
        else
          set_coefficient(pivot,
                          (get_coefficient(pivot) + get_coefficient(column.top())) % modulus);
        column.pop();
      }
      return (get_coefficient(pivot) == 0) ? -1 : pivot;
  #else
      while (!column.empty()) {
        pivot = column.top();
        column.pop();
        if (column.empty() || get_index(column.top()) != get_index(pivot)) return pivot;
        column.pop();
      }
      return -1;
  #endif
    }

    template <typename Column> diameter_entry_t get_pivot(Column& column) {
      diameter_entry_t result = pop_pivot(column);
      if (get_index(result) != -1) column.push(result);
      return result;
    }

    template <typename Column>
    diameter_entry_t init_coboundary_and_get_pivot(const diameter_entry_t simplex,
                                                   Column& working_coboundary, const index_t& dim,
                                                   entry_hash_map& pivot_column_index) {
      //static
      //simplex_coboundary_enumerator cofacets1(*this);
      bool check_for_emergent_pair = true;
      cofacet_entries.clear();
      cofacets1.set_simplex(simplex, dim);
      while (cofacets1.has_next()) {
        diameter_entry_t cofacet = cofacets1.next();
        if (get_diameter(cofacet) <= threshold) {
          cofacet_entries.push_back(cofacet);
          if (check_for_emergent_pair && (get_diameter(simplex) == get_diameter(cofacet))) {
            if ((pivot_column_index.find(get_entry(cofacet)) == pivot_column_index.end()) &&
                (get_index(get_zero_apparent_facet(cofacet, dim + 1)) == -1))
              return cofacet;
            check_for_emergent_pair = false;
          }
        }
      }
      for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
      return get_pivot(working_coboundary);
    }

    template <typename Column>
    void add_simplex_coboundary(const diameter_entry_t simplex, const index_t& dim,
                                Column& working_reduction_column, Column& working_coboundary) {
      //static
      //simplex_coboundary_enumerator cofacets2(*this);
      working_reduction_column.push(simplex);
      cofacets2.set_simplex(simplex, dim);
      while (cofacets2.has_next()) {
        diameter_entry_t cofacet = cofacets2.next();
        if (get_diameter(cofacet) <= threshold) working_coboundary.push(cofacet);
      }
    }

    template <typename Column>
    void add_coboundary(compressed_sparse_matrix<diameter_entry_t>& reduction_matrix,
                        const std::vector<diameter_index_t>& columns_to_reduce,
                        const size_t index_column_to_add, const coefficient_t factor,
                        const size_t& dim, Column& working_reduction_column,
                        Column& working_coboundary) {
      diameter_entry_t column_to_add(columns_to_reduce[index_column_to_add], factor);
      add_simplex_coboundary(column_to_add, dim, working_reduction_column, working_coboundary);

      for (diameter_entry_t simplex : reduction_matrix.subrange(index_column_to_add)) {
        set_coefficient(simplex, get_coefficient(simplex) * factor % modulus);
        add_simplex_coboundary(simplex, dim, working_reduction_column, working_coboundary);
      }
    }

    void compute_pairs(const std::vector<diameter_index_t>& columns_to_reduce,
                       entry_hash_map& pivot_column_index, const index_t dim) {

  #ifdef PRINT_PERSISTENCE_PAIRS
      std::cout << "persistence intervals in dim " << dim << ":" << std::endl;
  #endif

      compressed_sparse_matrix<diameter_entry_t> reduction_matrix;

  #ifdef INDICATE_PROGRESS
      std::chrono::steady_clock::time_point next = std::chrono::steady_clock::now() + time_step;
  #endif
      for (size_t index_column_to_reduce = 0; index_column_to_reduce < columns_to_reduce.size();
           ++index_column_to_reduce) {

        diameter_entry_t column_to_reduce(columns_to_reduce[index_column_to_reduce], 1);
        value_t diameter = get_diameter(column_to_reduce);

        reduction_matrix.append_column();

        std::priority_queue<diameter_entry_t, std::vector<diameter_entry_t>,
                            greater_diameter_or_smaller_index_comp<diameter_entry_t>>
            working_reduction_column, working_coboundary;

        diameter_entry_t e, pivot = init_coboundary_and_get_pivot(
                                column_to_reduce, working_coboundary, dim, pivot_column_index);

        while (true) {
  #ifdef INDICATE_PROGRESS
          if (std::chrono::steady_clock::now() > next) {
            std::cerr << clear_line << "reducing column " << index_column_to_reduce + 1
                      << "/" << columns_to_reduce.size() << " (diameter " << diameter << ")"
                      << std::flush;
            next = std::chrono::steady_clock::now() + time_step;
          }
  #endif
          if (get_index(pivot) != -1) {
            auto pair = pivot_column_index.find(get_entry(pivot));
            if (pair != pivot_column_index.end()) {
              entry_t other_pivot = pair->first;
              index_t index_column_to_add = pair->second;
              coefficient_t factor =
                  modulus - get_coefficient(pivot) *
                                multiplicative_inverse[get_coefficient(other_pivot)] %
                                modulus;

              add_coboundary(reduction_matrix, columns_to_reduce, index_column_to_add,
                             factor, dim, working_reduction_column, working_coboundary);

              pivot = get_pivot(working_coboundary);
            } else if (get_index(e = get_zero_apparent_facet(pivot, dim + 1)) != -1) {
              set_coefficient(e, modulus - get_coefficient(e));

              add_simplex_coboundary(e, dim, working_reduction_column, working_coboundary);

              pivot = get_pivot(working_coboundary);
            } else {

              value_t death = get_diameter(pivot);
  #ifdef PRINT_PERSISTENCE_PAIRS
              if (death > diameter * ratio) {
  #ifdef INDICATE_PROGRESS
                std::cerr << clear_line << std::flush;
  #endif
                std::cout << " [" << diameter << "," << death << ")" << std::endl;
              }
  #endif
              if (death > diameter * ratio)
                intervals[dim].emplace_back(diameter, death);

              pivot_column_index.insert({get_entry(pivot), index_column_to_reduce});

              while (true) {
                diameter_entry_t e = pop_pivot(working_reduction_column);
                if (get_index(e) == -1) break;
                assert(get_coefficient(e) > 0);
                reduction_matrix.push_back(e);
              }
              break;
            }
          } else {
  #ifdef PRINT_PERSISTENCE_PAIRS
  #ifdef INDICATE_PROGRESS
            std::cerr << clear_line << std::flush;
  #endif
            std::cout << " [" << diameter << ", )" << std::endl;
  #endif
            intervals[dim].emplace_back(diameter);
            break;
          }
        }
      }
  #ifdef INDICATE_PROGRESS
      std::cerr << clear_line << std::flush;
  #endif
    }

    std::vector<diameter_index_t> get_edges();

    void compute_barcodes() {
      std::vector<diameter_index_t> simplices, columns_to_reduce;

      compute_dim_0_pairs(simplices, columns_to_reduce);

      for (index_t dim = 1; dim <= dim_max; ++dim) {
        entry_hash_map pivot_column_index;
        pivot_column_index.reserve(columns_to_reduce.size());

        compute_pairs(columns_to_reduce, pivot_column_index, dim);

        if (dim < dim_max)
          assemble_columns_to_reduce(simplices, columns_to_reduce, pivot_column_index,
                                     dim + 1);
      }
    }
  };

  template <>
  simplex_coboundary_enumerator<compressed_lower_distance_matrix>::simplex_coboundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
                                const ripser<compressed_lower_distance_matrix>& _parent)
      : modulus(_parent.modulus), dist(_parent.dist),
        binomial_coeff(_parent.binomial_coeff), parent(_parent) {
    if (get_index(_simplex) != -1)
      parent.get_simplex_vertices(get_index(_simplex), _dim, parent.n, vertices.rbegin());
  }

  template<>
  simplex_coboundary_enumerator<compressed_lower_distance_matrix>::simplex_coboundary_enumerator(const ripser<compressed_lower_distance_matrix>& _parent) : modulus(_parent.modulus), dist(_parent.dist),
  binomial_coeff(_parent.binomial_coeff), parent(_parent) {}

  template<> void simplex_coboundary_enumerator<compressed_lower_distance_matrix>::set_simplex(const diameter_entry_t _simplex, const index_t _dim) {
    idx_below = get_index(_simplex);
    idx_above = 0;
    j = parent.n - 1;
    k = _dim + 1;
    simplex = _simplex;
    vertices.resize(_dim + 1);
    parent.get_simplex_vertices(get_index(_simplex), _dim, parent.n, vertices.rbegin());
  }

  template <> bool simplex_coboundary_enumerator<compressed_lower_distance_matrix>::has_next(bool all_cofacets) {
    return (j >= k && (all_cofacets || binomial_coeff(j, k) > idx_below));
  }

  template <> diameter_entry_t simplex_coboundary_enumerator<compressed_lower_distance_matrix>::next() {
    while ((binomial_coeff(j, k) <= idx_below)) {
      idx_below -= binomial_coeff(j, k);
      idx_above += binomial_coeff(j, k + 1);
      --j;
      --k;
      assert(k != -1);
    }
    value_t cofacet_diameter = get_diameter(simplex);
    for (index_t i : vertices) cofacet_diameter = std::max(cofacet_diameter, dist(j, i));
    index_t cofacet_index = idx_above + binomial_coeff(j--, k + 1) + idx_below;
    coefficient_t cofacet_coefficient =
        (k & 1 ? modulus - 1 : 1) * get_coefficient(simplex) % modulus;
    return diameter_entry_t(cofacet_diameter, cofacet_index, cofacet_coefficient);
  }

#if 0
  template<> void
  simplex_coboundary_enumerator<sparse_distance_matrix>::set_simplex(const diameter_entry_t _simplex, const index_t _dim) {
    idx_below = get_index(_simplex);
    idx_above = 0;
    k = _dim + 1;
    simplex = _simplex;
    vertices.resize(_dim + 1);
    parent.get_simplex_vertices(idx_below, _dim, parent.n, vertices.rbegin());

    neighbor_it.resize(_dim + 1);
    neighbor_end.resize(_dim + 1);
    for (index_t i = 0; i <= _dim; ++i) {
      auto v = vertices[i];
      neighbor_it[i] = dist.neighbors[v].rbegin();
      neighbor_end[i] = dist.neighbors[v].rend();
    }
  }

  template <>
  simplex_coboundary_enumerator<sparse_distance_matrix>::simplex_coboundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
                                const ripser<sparse_distance_matrix>& _parent)
      : modulus(_parent.modulus), dist(_parent.dist),
        binomial_coeff(_parent.binomial_coeff), parent(_parent) {
    if (get_index(_simplex) != -1) set_simplex(_simplex, _dim);
  }

  template <>
  simplex_coboundary_enumerator<sparse_distance_matrix>::simplex_coboundary_enumerator(const ripser<sparse_distance_matrix>& _parent)
        : modulus(_parent.modulus), dist(_parent.dist),
    binomial_coeff(_parent.binomial_coeff), parent(_parent) {}

  //template<> void simplex_coboundary_enumerator<compressed_lower_distance_matrix>::set_simplex(const diameter_entry_t _simplex, const index_t _dim) {


  template<> bool
  simplex_coboundary_enumerator<sparse_distance_matrix>::has_next(bool all_cofacets) {
    for (auto &it0 = neighbor_it[0], &end0 = neighbor_end[0]; it0 != end0; ++it0) {
      neighbor = *it0;
      for (size_t idx = 1; idx < neighbor_it.size(); ++idx) {
        auto &it = neighbor_it[idx], end = neighbor_end[idx];
        while (get_index(*it) > get_index(neighbor))
          if (++it == end) return false;
        if (get_index(*it) != get_index(neighbor))
          goto continue_outer;
        else
          neighbor = std::max(neighbor, *it);
      }
      while (k > 0 && vertices[k - 1] > get_index(neighbor)) {
        if (!all_cofacets) return false;
        idx_below -= binomial_coeff(vertices[k - 1], k);
        idx_above += binomial_coeff(vertices[k - 1], k + 1);
        --k;
      }
      return true;
    continue_outer:;
    }
    return false;
  }

  template <>
  diameter_entry_t
  simplex_coboundary_enumerator<sparse_distance_matrix>::next() {
    ++neighbor_it[0];
    value_t cofacet_diameter = std::max(get_diameter(simplex), get_diameter(neighbor));
    index_t cofacet_index = idx_above + binomial_coeff(get_index(neighbor), k + 1) + idx_below;
    coefficient_t cofacet_coefficient =
        (k & 1 ? modulus - 1 : 1) * get_coefficient(simplex) % modulus;
    return diameter_entry_t(cofacet_diameter, cofacet_index, cofacet_coefficient);
  }
#endif

  template <> inline std::vector<diameter_index_t> ripser<compressed_lower_distance_matrix>::get_edges() {
    std::vector<diameter_index_t> edges;
    std::vector<index_t> vertices(2);
    for (index_t index = binomial_coeff(n, 2); index-- > 0;) {
      get_simplex_vertices(index, 1, dist.size(), vertices.rbegin());
      value_t length = dist(vertices[0], vertices[1]);
      if (length <= threshold) edges.push_back({length, index});
    }
    return edges;
  }

  template <> inline std::vector<diameter_index_t> ripser<sparse_distance_matrix>::get_edges() {
    std::vector<diameter_index_t> edges;
    for (index_t i = 0; i < n; ++i)
      for (auto n : dist.neighbors[i]) {
        index_t j = get_index(n);
        if (i > j) edges.push_back({get_diameter(n), get_edge_index(i, j)});
      }
    return edges;
  }

  enum file_format {
    LOWER_DISTANCE_MATRIX,
    UPPER_DISTANCE_MATRIX,
    DISTANCE_MATRIX,
    POINT_CLOUD,
    DIPHA,
    SPARSE,
    BINARY
  };

  static const uint16_t endian_check(0xff00);
  static const bool is_big_endian = *reinterpret_cast<const uint8_t*>(&endian_check);

  template <typename T> T read(std::istream& input_stream) {
    T result;
    char* p = reinterpret_cast<char*>(&result);
    if (input_stream.read(p, sizeof(T)).gcount() != sizeof(T)) return T();
    if (is_big_endian) std::reverse(p, p + sizeof(T));
    return result;
  }

#if 0
  euclidean_distance_matrix read_point_cloud(std::istream& input_stream) {
    std::vector<std::vector<value_t>> points;

    std::string line;
    value_t value;
    while (std::getline(input_stream, line)) {
      std::vector<value_t> point;
      std::istringstream s(line);
      while (s >> value) {
        point.push_back(value);
        s.ignore();
      }
      if (!point.empty()) points.push_back(point);
      assert(point.size() == points.front().size());
    }

    euclidean_distance_matrix eucl_dist(std::move(points));
    index_t n = eucl_dist.size();
    std::cout << "point cloud with " << n << " points in dimension "
              << eucl_dist.points.front().size() << std::endl;

    return eucl_dist;
  }

  inline sparse_distance_matrix read_sparse_distance_matrix(std::istream& input_stream) {
    std::vector<std::vector<index_diameter_t>> neighbors;
    index_t num_edges = 0;

    std::string line;
    while (std::getline(input_stream, line)) {
      std::istringstream s(line);
      size_t i, j;
      value_t value;
      s >> i;
      s.ignore();
      s >> j;
      s.ignore();
      s >> value;
      s.ignore();
      if (i != j) {
        neighbors.resize(std::max({neighbors.size(), i + 1, j + 1}));
        neighbors[i].push_back({j, value});
        neighbors[j].push_back({i, value});
        ++num_edges;
      }
    }

    for (size_t i = 0; i < neighbors.size(); ++i)
      std::sort(neighbors[i].begin(), neighbors[i].end());

    return sparse_distance_matrix(std::move(neighbors), num_edges);
  }

  inline compressed_lower_distance_matrix read_lower_distance_matrix(std::istream& input_stream) {
    std::vector<value_t> distances;
    value_t value;
    while (input_stream >> value) {
      distances.push_back(value);
      input_stream.ignore();
    }

    return compressed_lower_distance_matrix(std::move(distances));
  }

  inline compressed_lower_distance_matrix read_upper_distance_matrix(std::istream& input_stream) {
    std::vector<value_t> distances;
    value_t value;
    while (input_stream >> value) {
      distances.push_back(value);
      input_stream.ignore();
    }

    return compressed_lower_distance_matrix(compressed_upper_distance_matrix(std::move(distances)));
  }

  inline compressed_lower_distance_matrix read_distance_matrix(std::istream& input_stream) {
    std::vector<value_t> distances;

    std::string line;
    value_t value;
    for (int i = 0; std::getline(input_stream, line); ++i) {
      std::istringstream s(line);
      for (int j = 0; j < i && s >> value; ++j) {
        distances.push_back(value);
        s.ignore();
      }
    }

    return compressed_lower_distance_matrix(std::move(distances));
  }

  inline compressed_lower_distance_matrix read_dipha(std::istream& input_stream) {
    if (read<int64_t>(input_stream) != 8067171840) {
      std::cerr << "input is not a Dipha file (magic number: 8067171840)" << std::endl;
      exit(-1);
    }

    if (read<int64_t>(input_stream) != 7) {
      std::cerr << "input is not a Dipha distance matrix (file type: 7)" << std::endl;
      exit(-1);
    }

    index_t n = read<int64_t>(input_stream);

    std::vector<value_t> distances;

    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (i > j)
          distances.push_back(read<double>(input_stream));
        else
          read<double>(input_stream);

    return compressed_lower_distance_matrix(std::move(distances));
  }

  inline compressed_lower_distance_matrix read_binary(std::istream& input_stream) {
    std::vector<value_t> distances;
    while (!input_stream.eof()) distances.push_back(read<value_t>(input_stream));
    return compressed_lower_distance_matrix(std::move(distances));
  }

  inline compressed_lower_distance_matrix read_file(std::istream& input_stream, const file_format format) {
    switch (format) {
    case LOWER_DISTANCE_MATRIX:
      return read_lower_distance_matrix(input_stream);
    case UPPER_DISTANCE_MATRIX:
      return read_upper_distance_matrix(input_stream);
    case DISTANCE_MATRIX:
      return read_distance_matrix(input_stream);
    case POINT_CLOUD:
      return read_point_cloud(input_stream);
    case DIPHA:
      return read_dipha(input_stream);
    default:
      return read_binary(input_stream);
    }
  }
#endif
}
