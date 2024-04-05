import masspcf as mpcf
from masspcf.random import noisy_sin, noisy_cos
from masspcf.plotting import plot as plotpcf
from masspcf.matrix_computations import pdist2

import matplotlib.pyplot as plt

M = 10
A = mpcf.zeros((2,M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
A[0,:] = noisy_sin((M,), n_points=100)
A[1,:] = noisy_cos((M,), n_points=15)

print('A')
print(pdist2(A[1,:]))
print('B')
print(pdist2(A[:,0]))
