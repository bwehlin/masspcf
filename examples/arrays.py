import masspcf as mpcf
from masspcf.random import noisy_sin, noisy_cos
from masspcf.plotting import plot as plotpcf

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

X = mpcf.zeros((10, 5, 4))
print(X[3, :, :].shape)
print(X[2:9:3, 1:, 2].shape) # [2,5,8] x [1,...,4] x [2]

M = 10
A = mpcf.zeros((2,M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
A[0,:] = noisy_sin((M,), n_points=100)
A[1,:] = noisy_cos((M,), n_points=15)

fig, ax = plt.subplots(1, 1, figsize=(6,2))

# Plot individual noisy sin/cos functions
for j in range(A.shape[1]): 
    plotpcf(A[0, j], ax=ax, color='b', linewidth=0.5, alpha=0.4)
for j in range(A.shape[1]): 
    plotpcf(A[1, j], ax=ax, color='r', linewidth=0.5, alpha=0.4)

# Means across first axis of 'A'
Aavg = mpcf.mean(A, dim=1)

# Plot means
plotpcf(Aavg[0], ax=ax, color='b', linewidth=2, label='$\\sin(2\\pi t)$')
plotpcf(Aavg[1], ax=ax, color='r', linewidth=2, label='$\\cos(2\\pi t)$')

plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend()

plt.savefig('means.png', dpi=300)

plt.show()

raise SystemExit

B = noisy_cos((10000,500), nPoints=30)
print(B.shape)

Bbar = mpcf.mean(B, 1)
print(Bbar.shape)

for j in range(100):
    plotpcf(Bbar[j], color='r', linewidth=2)
    
plt.show()
