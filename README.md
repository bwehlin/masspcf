# masspcf

Copyright 2024-2025 Bj&ouml;rn H. Wehlin

## Description

**masspcf** is a Python package with its backend written in C++ and CUDA for performing highly scalable computations involving piecewise constant functions (PCFs). The primary audience is practicioners within Topological Data Analysis (TDA) wanting to do statistical analysis on invariants such as *stable rank*, *Euler characteristic curves*, *Betti curves*, and so on.

The basic objects are **numpy**-like (multidimensional) arrays of PCFs, on which we support reductions such as taking averages across a dimension, etc. For 1-D arrays, we compute Lp distance matrices and L2 kernels that can then be used as input for, e.g., clustering, SVMs, and other machine learning algorithms.

## Installing

**masspcf** can be used on all three major platforms. Currently, users must install the CUDA toolkit locally (for Windows and Linux) to build **masspcf**. In addition, a compiler supporting C++ >=17 is required. 

The latest released version of the package can be obtained via PyPI.

`pip install masspcf`

For any issues installing or using **masspcf**, please file an issue on this GitHub repo.

## Citing

If you use **masspcf** in your research, please use the following BibTeX citation.

```
@misc{masspcf,
      title={Massively Parallel Computation of Similarity Matrices from Piecewise Constant Invariants}, 
      author={Bj\"{o}rn H. Wehlin},
      year={2024},
      eprint={2404.07183},
      archivePrefix={arXiv},
      primaryClass={stat.CO}
}
```

## Acknowledgment

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software
Program (WASP) funded by the Knut and Alice Wallenberg Foundation.