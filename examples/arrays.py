import masspcf as mpcf
from masspcf.random import noisy_sin
from masspcf.plotting import plot as plotpcf

import numpy as np

import matplotlib.pyplot as plt

z = mpcf.zeros((4,5))
print(z.shape)

A = noisy_sin(5)
print(A.shape)

z[0,:] = A

print(np.array(A.at([1])))
print(np.array(z.at([0, 0])))

#print(v)
