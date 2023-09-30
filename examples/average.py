from mpcf.pcf import Pcf, average, from_numpy
import numpy as np
import matplotlib.pyplot as plt

m = 3
n = 10

T = np.random.uniform(0.0, 6.0, size=(m, n))
T = np.sort(T, axis=1)
print(T)
V = np.sin(T) + 0.01 * np.random.randn(m, n)

fs = from_numpy(T, V)
print(fs)

plt.step(T.T, V.T, where='post')
plt.show()