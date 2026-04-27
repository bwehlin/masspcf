# masspcf — AI usage primer

A self-contained reference for an AI assistant that needs to write **user code that
calls masspcf** (not develop masspcf itself). Drop this file into the AI's context
before starting a project that uses masspcf.

---

## 1. What masspcf is

`masspcf` is a Python package with a C++20/CUDA backend for massively parallel
computation on **piecewise constant functions (PCFs)**. Primary audience: TDA
(Topological Data Analysis) practitioners doing statistical analysis on
invariants like *stable rank*, *Euler characteristic curves*, and *Betti curves*.

The core abstraction is a **NumPy-like N-dimensional tensor whose elements are
PCFs** (or related objects: point clouds, barcodes, distance/symmetric matrices,
plain floats/ints). Operations include reductions across dimensions, pairwise
$L^p$ distance matrices, $L^2$ kernels, persistent homology, and barcode
summaries.

Install with `pip install masspcf`. CUDA is auto-detected; CPU fallback is
always available.

---

## 2. The mental model

1. A `Pcf` is a single piecewise constant function defined by `(time, value)`
   breakpoints. The first breakpoint **must** have `t = 0`.
2. A `*Tensor` is an N-D array of those objects (or of floats / int / point
   clouds / barcodes / matrix-typed elements). Indexing and shape semantics
   mirror NumPy.
3. **dtype is a sentinel**, not a NumPy dtype. `mpcf.pcf32`, `mpcf.float64`, etc.
   are singleton tag objects. Compare with `x.dtype == mpcf.pcf32`, **never**
   `isinstance(x, mpcf.pcf32)`.
4. CPU/GPU dispatch is automatic. You usually don't think about it; tune via
   `mpcf.system.*` if needed.

---

## 3. Imports — what's at the top level

```python
import masspcf as mpcf
import numpy as np
```

`mpcf` exposes (from `masspcf/__init__.py`):

- **Single PCF**: `Pcf`
- **Tensor classes**: `PcfTensor`, `IntPcfTensor`, `FloatTensor`, `IntTensor`,
  `BoolTensor`, `PointCloudTensor`, `DistanceMatrix`, `DistanceMatrixTensor`,
  `SymmetricMatrix`, `SymmetricMatrixTensor` (+ `BarcodeTensor` from
  `mpcf.persistence`)
- **Tensor creation**: `zeros`, `stack`, `concatenate`, `split`, `array_split`
- **Dtype sentinels**: `pcf32`, `pcf64`, `pcf32i`, `pcf64i`, `float32`,
  `float64`, `int32`, `int64`, `uint32`, `uint64`, `boolean`, `pcloud32`,
  `pcloud64`, `barcode32`, `barcode64`, `distmat32`, `distmat64`, `symmat32`,
  `symmat64`
- **Operations**: `pdist`, `cdist`, `lp_distance`, `l2_kernel`, `lp_norm`,
  `mean`, `max_time`, `allclose`, `iterate_rectangles`
- **I/O**: `save`, `load`, `from_serial_content`
- **Submodules**: `mpcf.random`, `mpcf.system`, `mpcf.persistence`,
  `mpcf.plotting`, `mpcf.point_process`

---

## 4. Building a single PCF

`Pcf(arr, dtype=None)` accepts an `ndarray` of shape `(n, 2)` whose **rows are
`[time, value]` pairs**, an existing `Pcf` (copy), or a Python list of rows.
The dtype is inferred from the array's NumPy dtype unless given explicitly.

```python
# A PCF that equals 1 on [0,2), 3 on [2,5), 0 on [5, +inf)
f = mpcf.Pcf(np.array([[0.0, 1.0],
                       [2.0, 3.0],
                       [5.0, 0.0]]))

f(1.0)   # -> 1.0  (callable)
f(3.5)   # -> 3.0
f.size   # number of breakpoints

g = f * 2          # pointwise scale
h = f + g          # pointwise add
s = f ** 0.5       # pointwise sqrt

f.to_numpy()       # back to (n, 2) array
f.astype(mpcf.pcf64)
```

Constraints:
- First time **must** be `0.0`; times must be strictly increasing.
- Supported dtypes: `pcf32`, `pcf64` (float values), `pcf32i`, `pcf64i`
  (integer values — for invariants that are integer-valued; `pdist` / `lp_norm`
  reject these, see §11).

---

## 5. Tensors of PCFs (and friends)

### Allocation

```python
# A 1-D tensor holding 3 PCFs (default dtype is pcf32)
X = mpcf.zeros((3,))
X[0] = f1
X[1] = f2
X[2] = f3

# Higher-rank PCF tensors
A = mpcf.zeros((4, 100), dtype=mpcf.pcf64)
A[0, 7] = some_pcf

# Float/int/point-cloud/bool tensors
F = mpcf.zeros((10, 10), dtype=mpcf.float32)
P = mpcf.zeros((5,),     dtype=mpcf.pcloud64)
P[0] = np.random.rand(50, 3)        # any (n_points, dim) array
B = mpcf.zeros((5,),     dtype=mpcf.barcode32)  # usually produced by ripser
```

### Indexing

NumPy-style. `X[i, j]` returns a single `Pcf` (when fully indexed) or a tensor
view (when sliced). Slices are **views**, not copies.

```python
X = mpcf.zeros((10, 5, 4))
X[3, :, :].shape       # (5, 4)
X[2:9:3, 1:, 2].shape  # (3, 4)
```

### Stacking / splitting

```python
mpcf.stack([X1, X2, X3], axis=0)
mpcf.concatenate([X1, X2], axis=0)
mpcf.split(X, n_chunks, axis=0)
mpcf.array_split(X, n_chunks, axis=0)
```

---

## 6. Random / synthetic data — `mpcf.random`

```python
from masspcf.random import noisy_sin, noisy_cos, Generator, seed

seed(42)                 # global RNG
rng = Generator(42)      # local RNG (pass via generator=)

# noisy_sin(shape, n_points=20, dtype=pcf32, generator=None) -> PcfTensor
sines  = noisy_sin((200,),    n_points=100)   # 200 PCFs
cosines = noisy_cos((10, 50), n_points=30)    # shape (10, 50)
```

Spatial Poisson point clouds (in `mpcf.point_process.poisson`):

```python
from masspcf.point_process.poisson import sample_poisson
pts = sample_poisson((100,), dim=2, rate=1.0, lo=0.0, hi=1.0,
                     dtype=mpcf.pcloud64)   # PointCloudTensor of shape (100,)
```

---

## 7. Reductions, norms, distances, kernels

### Reductions (`mpcf.reductions`)
```python
mpcf.mean(A, dim=1)        # PcfTensor reduced along axis 1
mpcf.max_time(A, dim=0)    # FloatTensor: max breakpoint time
```

### Norms (`mpcf.norms`)
```python
norms = mpcf.lp_norm(X, p=1)            # FloatTensor, same shape as X
norms = mpcf.lp_norm(X, p=2.5, verbose=True)
```

### Distances (`mpcf.distance`)
```python
# Single pair
d = mpcf.lp_distance(f, g, p=1)         # float

# Pairwise within a 1-D PcfTensor (must be 1-D!)
D = mpcf.pdist(X, p=1)                  # DistanceMatrix (compressed)

# All-pairs between two tensors (any rank)
M = mpcf.cdist(X, Y, p=1)               # FloatTensor of shape (*X.shape, *Y.shape)
```

`pdist` and `l2_kernel` require a **1-D** PCF tensor.

### L2 kernel (`mpcf.inner_product`)
```python
K = mpcf.l2_kernel(X)                   # SymmetricMatrix, useful for SVMs
```

### Comparison
```python
mpcf.allclose(a, b, atol=1e-8, rtol=1e-5)  # works on FloatTensor /
                                           # DistanceMatrix / SymmetricMatrix
```

`DistanceMatrix` / `SymmetricMatrix` are **compressed** representations. Convert
to a dense `FloatTensor` / NumPy array via their `.to_numpy()` / `.to_dense()`
methods if you need the full matrix.

---

## 8. Persistent homology (`mpcf.persistence`)

```python
from masspcf.persistence import (
    compute_persistent_homology,
    barcode_to_stable_rank,
    barcode_to_betti_curve,
    barcode_to_accumulated_persistence,
    ComplexType, DistanceType,
)

# Input: a PointCloudTensor, a DistanceMatrix(Tensor), or a FloatTensor
pts = mpcf.zeros((1,), dtype=mpcf.pcloud32)
pts[0] = np.random.rand(50, 3).astype(np.float32)

# Output shape: (*pts.shape, max_dim+1)  -- last axis is homology dimension
barcodes = compute_persistent_homology(
    pts,
    max_dim=1,
    distance_type=DistanceType.Euclidean,
    complex_type=ComplexType.VietorisRips,
    reduced=False,
)

sr  = barcode_to_stable_rank(barcodes)        # Pcf or PcfTensor
bc  = barcode_to_betti_curve(barcodes)
apf = barcode_to_accumulated_persistence(barcodes)

# Index the homology dimension on the last axis:
sr_h1 = sr[..., 1]         # H_1 stable ranks
```

---

## 9. Plotting (`mpcf.plotting`)

```python
import matplotlib.pyplot as plt
from masspcf.plotting import plot, plot_barcode

plot(f)                              # one PCF
plot(X[0])                           # one element of a tensor
plot(X, ax=ax, color="b", alpha=0.4) # plot a 1-D PcfTensor
plot_barcode(barcodes[0])
plt.show()
```

`plot` wraps `matplotlib.step` and accepts `ax=`, `auto_label=`, `max_time=`,
plus any matplotlib kwargs.

---

## 10. I/O

```python
mpcf.save(X, "out.mpcf")
X = mpcf.load("out.mpcf")
```

Works on any masspcf tensor / matrix object. File handles also accepted.

---

## 11. Device and runtime control (`mpcf.system`)

```python
mpcf.system.force_cpu(True)             # CPU only
mpcf.system.limit_cpus(8)               # cap thread count
mpcf.system.limit_gpus(1)               # cap GPU count
mpcf.system.set_cuda_threshold(1000)    # use GPU once N(PCFs) exceeds this
                                        # (default ~500)
mpcf.system.set_device_verbose(True)    # print CPU/GPU dispatch decisions
mpcf.system.build_type()                # "CUDA" or "CPU"
```

CUDA is used automatically once problem size crosses the threshold.

---

## 12. Gotchas (read this before writing code)

1. **Pcf row format is `[time, value]`**, shape `(n, 2)`. Times must start at
   `0` and be strictly increasing.
2. **`dtype`s are sentinels**, not NumPy dtypes or classes. Use
   `x.dtype == mpcf.pcf32`, not `isinstance`.
3. **`pdist` and `l2_kernel` need a 1-D tensor.** For an `(N, M)` `PcfTensor`,
   reshape / index down to 1-D first, or use `cdist` / `mean(..., dim=k)`.
4. **Integer PCFs (`pcf32i`, `pcf64i`) don't support `lp_distance` / `pdist` /
   `lp_norm`.** Cast with `f.astype(mpcf.pcf64)` first.
5. **`DistanceMatrix` and `SymmetricMatrix` are compressed**, not dense
   `FloatTensor`s. Use their conversion methods to get NumPy arrays.
6. **Slicing returns views**, not copies. Mutate carefully; use `.copy()` to
   detach.
7. **Persistent homology output has a trailing dimension** of size `max_dim+1`
   indexing the homology degree. Always grab `[..., k]` for $H_k$.
8. **Always import as `import masspcf as mpcf`** — many third-party tutorials
   and old example files use the legacy name `mpcf` directly (e.g.
   `from mpcf.pcf import Pcf` or `force_cpu` at the top level). Those are
   stale; only the surface in `masspcf/__init__.py` (this guide) is current.
9. **GPU vs CPU output should match within float tolerance**, not exactly.
   Use `mpcf.allclose` with sensible tolerances when cross-validating.

---

## 13. Canonical end-to-end snippets

### A. Mean of noisy sin/cos PCFs
```python
import masspcf as mpcf
from masspcf.random import noisy_sin, noisy_cos
from masspcf.plotting import plot
import matplotlib.pyplot as plt

A = mpcf.zeros((2, 10))
A[0, :] = noisy_sin((10,), n_points=100)
A[1, :] = noisy_cos((10,), n_points=15)

avg = mpcf.mean(A, dim=1)        # shape (2,)

fig, ax = plt.subplots()
for j in range(A.shape[1]):
    plot(A[0, j], ax=ax, color="b", linewidth=0.5, alpha=0.4)
plot(avg[0], ax=ax, color="b", linewidth=2, label=r"$\sin$ mean")
ax.legend(); plt.show()
```

### B. Pairwise L1 distances → kernel
```python
X = mpcf.zeros((4,))
X[0] = f1; X[1] = f2; X[2] = f3; X[3] = f4
D = mpcf.pdist(X, p=1)           # DistanceMatrix
K = mpcf.l2_kernel(X)            # SymmetricMatrix (use as kernel for SVM, etc.)
D_dense = D.to_numpy()
```

### C. Point cloud → persistent homology → stable rank → distance
```python
import numpy as np, masspcf as mpcf
from masspcf.persistence import (
    compute_persistent_homology, barcode_to_stable_rank,
)

N = 50                                  # number of point clouds
pts = mpcf.zeros((N,), dtype=mpcf.pcloud32)
for i in range(N):
    pts[i] = np.random.rand(80, 3).astype(np.float32)

barcodes = compute_persistent_homology(pts, max_dim=1)   # (N, 2)
sr_h1    = barcode_to_stable_rank(barcodes)[..., 1]      # (N,) PcfTensor

D = mpcf.pdist(sr_h1, p=2)              # statistical distance between subjects
```

### D. Save / load tensors
```python
mpcf.save(sr_h1, "stable_ranks.mpcf")
sr_h1 = mpcf.load("stable_ranks.mpcf")
```

---

## 14. Where to look when this guide is not enough

- Quickstart: `docs/quickstart.rst`
- User guide: `docs/userguide.rst` and the topical pages
  (`tensors.rst`, `distances.rst`, `persistence.rst`, `plotting.rst`, ...)
- Working examples: `examples/` (note: `examples/dist.py` uses a stale API —
  prefer `examples/arrays.py`, `examples/average.py`,
  `examples/compute_homology.py`, `examples/l1_norm.py`,
  `examples/plotpcfs.py`, `examples/userguide/`)
- Source of truth for what's public: `masspcf/__init__.py`
