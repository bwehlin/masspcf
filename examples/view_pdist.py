import masspcf as mpcf
from masspcf.random import noisy_sin, noisy_cos
from masspcf.plotting import plot as plotpcf
from masspcf.matrix_computations import pdist
import numpy as np
import matplotlib.pyplot as plt
from masspcf.plotting import plot as plotpcf
from masspcf.array import max_time

import matplotlib.pyplot as plt

M = 10
A = mpcf.zeros((2,M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
A[0,:] = noisy_sin((M,), n_points=100)
A[1,:] = noisy_cos((M,), n_points=15)

print('A')
print(pdist(A[1,:]))
print('B')
print(pdist(A[:,0]))

f = mpcf.Pcf(np.array([[0., 5.], [2., 3.], [5., 0.]]), dtype=mpcf.float32)
g = mpcf.Pcf(np.array([[0., 2.], [4., 7.], [8., 1.], [9., 0.]]), dtype=mpcf.float32)
h = mpcf.Pcf(np.array([[0, 4], [2, 3], [3, 1], [5, 0]]), dtype=mpcf.float32)
k = mpcf.Pcf(np.array([[0, 2], [6, 1], [7, 0]]), dtype=mpcf.float32)

B = mpcf.zeros((4,))
B[0] = f
B[1] = g
B[2] = h
B[3] = k


print(pdist(B))
print(B.shape)
print(len(B.shape))

print(np.array(max_time(B, 0)))

plotpcf(B)
plt.legend()
#plotpcf(g)
plt.show()

