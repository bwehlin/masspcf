from mpcf.pcf import Pcf, average
import numpy as np
import matplotlib.pyplot as plt
import timeit

m = 3000 # Number of PCFs
n = 200 # Number of time points in each PCF

T = np.random.uniform(0.0, 6.0, size=(m, n))
T = np.sort(T, axis=1)
V = np.sin(T) + 0.01 * np.random.randn(m, n)

#plt.step(T.T, V.T, where='post')

fs = [None]*m
for i in range(m):
    X = np.vstack((T[i,:], V[i,:]))
    fs[i] = Pcf(X)


print(timeit.timeit('average(fs)', globals=globals(), number=10))

favg = average(fs)
fnp = favg.to_numpy()

plt.step(fnp[0,:], fnp[1,:], where='post', linewidth=3)
plt.show()
