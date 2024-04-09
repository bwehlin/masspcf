import masspcf as mpcf
import numpy as np

TV = np.array([[0, 4], [2, 3], [3, 1], [5, 0]])
f = mpcf.Pcf(TV)

print(f)

######

fMatrix = np.array(f)
print(fMatrix)

######

from masspcf.plotting import plot as plotpcf
import matplotlib.pyplot as plt

plotpcf(f)

plt.xlabel('t')
plt.ylabel('f(t)')

plt.savefig('ugfig_plot_single.png')

############# Multidimensional arrays #############

Z = mpcf.zeros((10, 5, 4))
print(Z.shape)

######

Z[3, :, :] # shape = (5,4), indices [3] x [0,...,4] x [0,...,3]
Z[2:9:3, 1:, 2] # shape = (3,4), indices [2,5,8] x [1,...,4] x [2]

######

from masspcf.random import noisy_sin, noisy_cos

M = 10 # Number of PCFs for each case
A = mpcf.zeros((2,M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
A[0,:] = noisy_sin((M,), n_points=100)
A[1,:] = noisy_cos((M,), n_points=15)

fig, ax = plt.subplots(1, 1, figsize=(6,2))

# Plot individual noisy sin/cos functions
# masspcf can plot one-dimensional arrays (views) of PCFs in a single line
plotpcf(A[0,:], ax=ax, color='b', linewidth=0.5, alpha=0.4)
plotpcf(A[1,:], ax=ax, color='r', linewidth=0.5, alpha=0.4)

# Means across first axis of 'A'
Aavg = mpcf.mean(A, dim=1)

# Plot means
plotpcf(Aavg[0], ax=ax, color='b', linewidth=2, label='sin')
plotpcf(Aavg[1], ax=ax, color='r', linewidth=2, label='cos')

ax.set_xlabel('t [2 pi]')
ax.set_ylabel('f(t)')
ax.legend()
fig.savefig('ugfig_noisy_means.png')

############# Matrix computations #############

f1 = mpcf.Pcf(np.array([[0., 5.], [2., 3.], [5., 0.]]))
f2 = mpcf.Pcf(np.array([[0., 2.], [4., 7.], [8., 1.], [9., 0.]]))
f3 = mpcf.Pcf(np.array([[0, 4], [2, 3], [3, 1], [5, 0]]))
f4 = mpcf.Pcf(np.array([[0, 2], [6, 1], [7, 0]]))

X = mpcf.Array([f1, f2, f3, f4])

print(mpcf.pdist(X))
print(mpcf.pdist(X, p=3.5))

print(mpcf.l2_kernel(X))