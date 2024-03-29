import masspcf as mpcf
from masspcf.random import noisy_sin
from masspcf.plotting import plot as plotpcf

import numpy as np

import matplotlib.pyplot as plt

z = mpcf.zeros((3, 4, 6))
print(z.shape())

#print(z[:,:2])
v = z[0, 1:3, :]


print(v.shape())
print(v.data)
v1 = v[0,1:3]
print(v1.shape())
print(v1.data)
v2 = v1[1:2]
print(v2.data)

A = noisy_sin(10)

print(np.array(A.at([0])))

plotpcf(A.at([0]))
plotpcf(A.at([1]))

plotpcf(mpcf.average([A.at([0]), A.at([1])]))

#print(v)
