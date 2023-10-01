from mpcf.pcf import Pcf, average, mem_average, st_average, matrix_l1_dist
import numpy as np
import matplotlib.pyplot as plt
import timeit

m = 50000 # Number of PCFs
n = 10 # Number of time points in each PCF

#m = 32
#n = 5

T = np.random.uniform(0.0, 6.0, size=(m, n))
T = np.sort(T, axis=1)
V = np.sin(T) + 0.01 * np.random.randn(m, n)

#plt.step(T.T, V.T, where='post')

fs = [None]*m
for i in range(m):
    X = np.vstack((T[i,:], V[i,:]))
    fs[i] = Pcf(X)

favg = average(fs)
fnp = favg.to_numpy()
print(fnp.shape)

#print(timeit.timeit('average(fs)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,2)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,3)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,4)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,8)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,16)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,32)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,64)', globals=globals(), number=10))
print(timeit.timeit('mem_average(fs,128)', globals=globals(), number=10))
#mem_average(fs)
raise SystemExit

#print(timeit.timeit('st_average(fs)', globals=globals(), number=10))

plt.step(fnp[0,:], fnp[1,:], where='post', linewidth=1)

favg2 = mem_average(fs)
print(matrix_l1_dist([favg, favg2]))


fnp = favg2.to_numpy()
print(fnp.shape)
plt.step(fnp[0,:], fnp[1,:], '-.', where='post', linewidth=1)

plt.show()
