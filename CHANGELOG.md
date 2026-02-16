# 0.4.0

* The `NdArray` class has been replaced with `Tensor`. The new version is purpose-built for **masspcf**. This means we are dropping the dependency on **xtensor** and **xtl**. The main reason for this change is so to make development more flexible and efficient. On the **Python** side, things should work the same as in previous versions (please report any inconsistencies/bugs related to the switch!).

## Breaking changes

* `Pcf`s can now be constructed from `numpy.array`s only of size `n x 2` (used to support also `2 x n`). This is to avoid ambiguities with `2x2` arrays.