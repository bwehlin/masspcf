/*
 Ripser++: accelerated Vietoris-Rips persistence barcodes computation with GPU

 MIT License

 Copyright (c) 2019, 2020 Simon Zhang, Mengbai Xiao, Hao Wang

 Python Bindings Contributors:
 Birkan Gokbag
 Ryan DeMilt

 Copyright (c) 2015-2019 Ripser codebase, written by Ulrich Bauer

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

#include <mpcf/cuda/cuda_util.cuh>

//#define INDICATE_PROGRESS//DO NOT UNCOMMENT THIS IF YOU WANT TO LOG PROFILING NUMBERS FROM stderr TO FILE
//#define PRINT_PERSISTENCE_PAIRS//print out all persistence paris to stdout
//#define CPUONLY_ASSEMBLE_REDUCTION_MATRIX//do full matrix reduction on CPU with the sparse coefficient matrix V
#define ASSEMBLE_REDUCTION_SUBMATRIX//do submatrix reduction with the sparse coefficient submatrix of V
//#define PROFILING
//#define COUNTING
// Upstream Ripser++ used phmap (greg7mdp/parallel-hashmap) because its
// `find_or_prepare_insert` is thread-safe, which matters when upstream's
// OpenMP sections shared the pivot map across threads. We parallelize at
// the outer (item-level) granularity instead -- each ripser instance
// owns its own pivot map -- so the single-threaded google dense_hash_map
// path is equivalent and avoids nvcc ICEs triggered by phmap's type-trait
// template chain on certain nvcc/libstdc++ combos (notably nvcc 12.9 +
// libstdc++13). No measured perf difference from keeping phmap.
#define USE_GOOGLE_HASHMAP
#define PYTHON_BARCODE_COLLECTION

//#define CPUONLY_SPARSE_HASHMAP//WARNING: MAY NEED LOWER GCC VERSION TO RUN, TESTED ON: NVCC VERSION 9.2 WITH GCC VERSIONS >=5.3.0 AND <=7.3.0

#define MIN_INT64 (-9223372036854775807-1)
#define MAX_INT64 (9223372036854775807)
#define MAX_FLOAT (340282346638528859811704183484516925440.000000)


#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <sparsehash/dense_hash_map>
#ifdef USE_PHASHMAP
#include <parallel_hashmap/phmap.h>
#endif

#include <mpcf/cuda/cuda_device_array.cuh>
#include <mpcf/cuda/cuda_stream.cuh>
#include <mpcf/executor.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <mpcf/cuda/cuda_async_memory_resource.cuh>
#ifdef CPUONLY_SPARSE_HASHMAP
#include <sparsehash/sparse_hash_map>
template <class Key, class T> class hash_map : public google::sparse_hash_map<Key, T> {
public:
    explicit hash_map() : google::sparse_hash_map<Key, T>() {
        }
    inline void reserve(size_t hint) { this->resize(hint); }
};
#endif

#ifndef CPUONLY_SPARSE_HASHMAP
template <class Key, class T> class hash_map : public google::dense_hash_map<Key, T> {
public:
    explicit hash_map() : google::dense_hash_map<Key, T>() {
        this->set_empty_key(-1);
    }
    inline void reserve(size_t hint) { this->resize(hint); }
};
#endif

#ifdef INDICATE_PROGRESS
static const std::chrono::milliseconds time_step(40);
static const std::string clear_line("\r\033[K");
#endif

// -----------------------------------------------------------------------------
// mpcf facade
//
// The upstream Ripser++ port below used a file-scope `typedef float value_t`,
// which silently downcast float64 inputs to float32 on the GPU path. To keep
// double precision end-to-end without refactoring the ~3400-line port into
// templates, the value_t-dependent body lives in `ripserpp_impl.inc` and is
// #included twice here under two detail namespaces (`detail_f32`, `detail_f64`)
// with different `using value_t = ...`. Each detail namespace gets its own
// instantiation of the entire port (structs, kernels, `ripser` class, helpers)
// at its chosen precision. The public entry points dispatch by T.
// -----------------------------------------------------------------------------

#include <mpcf/persistence/ripserpp/ripserpp.hpp>

#include <type_traits>

namespace mpcf::ph::ripserpp
{

namespace detail_f32 {
  using value_t = float;
  #include "ripserpp_impl.inc"
}

namespace detail_f64 {
  using value_t = double;
  #include "ripserpp_impl.inc"
}

template <typename T>
void compute_barcodes_pcloud(
    const PointCloud<T>& points,
    std::size_t maxDim,
    std::vector<std::vector<PersistencePair<T>>>& out,
    mpcf::Executor& exec,
    Diagnostics* diag,
    bool parallel_inner_loops)
{
  if constexpr (std::is_same_v<T, float>) {
    detail_f32::compute_barcodes_pcloud_impl<T>(points, maxDim, out, exec, diag,
                                                parallel_inner_loops);
  } else {
    detail_f64::compute_barcodes_pcloud_impl<T>(points, maxDim, out, exec, diag,
                                                parallel_inner_loops);
  }
}

template <typename T>
void compute_barcodes_distmat(
    const DistanceMatrix<T>& dmat,
    std::size_t maxDim,
    std::vector<std::vector<PersistencePair<T>>>& out,
    mpcf::Executor& exec,
    Diagnostics* diag,
    bool parallel_inner_loops)
{
  if constexpr (std::is_same_v<T, float>) {
    detail_f32::compute_barcodes_distmat_impl<T>(dmat, maxDim, out, exec, diag,
                                                 parallel_inner_loops);
  } else {
    detail_f64::compute_barcodes_distmat_impl<T>(dmat, maxDim, out, exec, diag,
                                                 parallel_inner_loops);
  }
}

template void compute_barcodes_pcloud<float>(const PointCloud<float>&, std::size_t,
                                             std::vector<std::vector<PersistencePair<float>>>&,
                                             mpcf::Executor&, Diagnostics*, bool);
template void compute_barcodes_pcloud<double>(const PointCloud<double>&, std::size_t,
                                              std::vector<std::vector<PersistencePair<double>>>&,
                                              mpcf::Executor&, Diagnostics*, bool);

template void compute_barcodes_distmat<float>(const DistanceMatrix<float>&, std::size_t,
                                              std::vector<std::vector<PersistencePair<float>>>&,
                                              mpcf::Executor&, Diagnostics*, bool);
template void compute_barcodes_distmat<double>(const DistanceMatrix<double>&, std::size_t,
                                               std::vector<std::vector<PersistencePair<double>>>&,
                                               mpcf::Executor&, Diagnostics*, bool);

}  // namespace mpcf::ph::ripserpp
