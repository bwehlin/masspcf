## 0.4.0

Major rewrite of the core data structures and significant expansion of the API.

### New features

* **Tensor type system** — `NdArray` replaced with a family of purpose-built tensor classes: `PcfTensor`, `IntPcfTensor`, `FloatTensor`, `IntTensor`, `BoolTensor`, `PointCloudTensor`, `DistanceMatrixTensor`, `SymmetricMatrixTensor`, `BarcodeTensor`. Each has a corresponding `dtype` sentinel.
* **NumPy-like tensor operations** — `reshape`, `transpose`/`.T`, `squeeze`, `expand_dims`, `swapaxes`, `concatenate`, `stack`, `split`, `array_split`, `astype`, iteration, `ndim`, `size`, `len()`.
* **Advanced indexing** — slicing with negative strides, multi-axis boolean masking, mixed integer+boolean indexing, broadcasting assignment.
* **Arithmetic** — element-wise `+`, `-`, `*`, `/`, `//`, `**`, unary `-` on numeric tensors; broadcasting support.
* **Persistence module** — `compute_persistent_homology` (Ripser), `Barcode`, `BarcodeTensor`, and barcode summaries: `barcode_to_stable_rank`, `barcode_to_betti_curve`, `barcode_to_accumulated_persistence`.
* **`DistanceMatrix` and `SymmetricMatrix`** — dedicated types with I/O support, `from_dense` and `to_dense` for NumPy interop.
* **Tensor I/O** — `save`/`load` for all tensor types; `from_serial_content` for in-memory deserialization.
* **PCF evaluation** — evaluate PCFs at given time points.
* **Plotting** — built-in plotting for barcodes (matplotlib).
* **`lp_distance`** — scalar Lp distance between two individual PCFs.
* **`cdist`** — cross-distance matrices between two collections of PCFs.
* **`allclose`** — free function for element-wise approximate equality (FloatTensor, DistanceMatrix, SymmetricMatrix).
* **`pickle` support** — all tensor types can be pickled and unpickled.
* **Deterministic random generation** — seedable `Generator` for reproducible output across threads.
* **`point_process` submodule** — `sample_poisson` for sampling spatial Poisson point processes.

### Breaking changes

* `pdist` returns a `DistanceMatrix` instead of a NumPy array. Call `.to_dense()` to get the previous behavior.
* `l2_kernel` returns a `SymmetricMatrix` instead of a NumPy array. Call `.to_dense()` to get the previous behavior.
* `NdArray` replaced with `Tensor` classes (dropped **xtensor**/**xtl** dependencies).
* `Pcf` construction now only accepts `n x 2` arrays (not `2 x n`) to avoid ambiguity with `2x2` arrays.
* I/O format bumped to version 2.
* Requires C++20 (was C++17). Minimum Python version is 3.10.

### Infrastructure

* CUDA backend reworked: auto-detection, pip-installed CUDA toolkit support (`nvidia.cu12`/`nvidia.cu13`), version-specific modules.
* Wheels built for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64).
* CUDA matrix integration refactored into modular components.
* GPU occupancy floor added to block scheduler.