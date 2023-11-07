from mpcf.pcf import average, Pcf, matrix_l1_dist
import mpcf.plotting as mplt

import numpy as np
import matplotlib.pyplot as plt

import timeit

m = 300 # Number of PCFs
n = 5000 # Number of time points in each PCF

print('Gen')
T = np.random.uniform(0.0, 3.0, size=(m, n-1))
T = np.sort(T, axis=1)
T = np.hstack((T, 1.1 * np.max(np.max(T)) * np.ones((m, 1))))

V = np.sin(T) + 0.01 * np.random.randn(m, n)
V[:,-1] = 0

T = T.astype(np.float32)
V = V.astype(np.float32)

fs = [None]*m
for i in range(m):
    X = np.vstack((T[i,:], V[i,:]))
    fs[i] = Pcf(X)

oldAvg = average(fs, False)
newAvg = average(fs, True)

print(matrix_l1_dist([oldAvg, newAvg]))

mplt.plot(oldAvg)
mplt.plot(newAvg)

def bench(newCode):
    times = timeit.repeat(f'average(fs, {True if newCode else False})', globals=globals(), number=1, repeat=300)
    avtime = np.average(times)
    sttime = np.std(times)
    print(f'avg: {avtime}s, std: {sttime}s')
    return avtime

print(newAvg.to_numpy().shape)
print(oldAvg.to_numpy().shape)
print(newAvg.to_numpy())
print(oldAvg.to_numpy())

anew = bench(True)
aold = bench(False)

print(f'Speedup: {aold/anew}')


#plt.show()



