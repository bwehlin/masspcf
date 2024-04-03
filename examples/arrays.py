import masspcf as mpcf
from masspcf.random import noisy_sin, noisy_cos
from masspcf.plotting import plot as plotpcf
from masspcf.array import mean

import numpy as np

import matplotlib.pyplot as plt

n = 10

A = mpcf.zeros((2,n))

A[0,:] = noisy_sin((n,), nPoints=50)
A[1,:] = noisy_cos((n,), nPoints=50)

for j in range(A.shape[1]):
    plotpcf(A[0, j], color='b', linewidth=0.5)

for j in range(A.shape[1]):
    plotpcf(A[1, j], color='r', linewidth=0.5)

Abar = mean(A, 1)

plotpcf(Abar[0], color='b', linewidth=2)
plotpcf(Abar[1], color='r', linewidth=2)

plt.show()

