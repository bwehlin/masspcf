# 0.4.0

* The `NdArray` class has been replaced with `Tensor`. The new version is purpose-built for **masspcf**. This means we are dropping the dependency on **xtensor** and **xtl**. The main reason for this change is so to make development more flexible and efficient. On the **Python** side, things should work the same as in previous versions (please report any inconsistencies/bugs related to the switch!).

## Breaking changes

* `Pcf`s can now be constructed from `numpy.array`s only of size `n x 2` (used to support also `2 x n`). This is to avoid ambiguities with `2x2` arrays.
* The `dtype`s `mpcf.float32` and `mpcf.float64` have been replaced by `mpcf.pcf32` and `mpcf.pcf64`, respectively. Users that rely on the old `dtype`s may see deprecation warnings, and the `dtype`s may be removed in a future release. This change is to support tensors of different types, including numeric types (see `mpcf.f32` and `mpcf.f64`). 