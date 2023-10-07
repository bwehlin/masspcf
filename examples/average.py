print('1')

from mpcf.pcf import Pcf, average, mem_average, st_average, matrix_l1_dist, back_average, async_matrix_l1_dist

print('2')

import numpy as np
import matplotlib.pyplot as plt
import timeit

m = 5000 #00 # Number of PCFs
n = 1000 # Number of time points in each PCF

print('3')

#m = 32
#n = 5

T = np.random.uniform(0.0, 6.0, size=(m, n))
T = np.sort(T, axis=1)
V = np.sin(T) + 0.01 * np.random.randn(m, n)

#plt.step(T.T, V.T, where='post')

print('4')

fs = [None]*m
for i in range(m):
    X = np.vstack((T[i,:], V[i,:]))
    fs[i] = Pcf(X)

def benchmark(stmt, label=''):
    times = timeit.repeat(stmt, globals=globals(), repeat=5, number=1)
    print(f'{label} Mean: {np.mean(times)*1000}ms, sd: {np.std(times)*1000}ms')

#fnp = favg.to_numpy()
#print(fnp.shape)

print('Running CUDA stuff')
print(async_matrix_l1_dist(fs))

raise SystemExit

for i in [2, 10, 50, 100, 500, 1000, 5000, 10000]:
    benchmark(f'average(fs[:i+1])',     label=f'async n={i}')
    benchmark(f'mem_average(fs[:i+1],2)', label=f'st    n={i}')
#print(timeit.repeat('average(fs)', globals=globals(), repeat=100, number=1))

#print(timeit.timeit('average(fs)', globals=globals(), number=100))
#print(timeit.timeit('mem_average(fs,2)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,3)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,4)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,8)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,16)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,32)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,64)', globals=globals(), number=10))
#print(timeit.timeit('mem_average(fs,128)', globals=globals(), number=10))
#mem_average(fs)
raise SystemExit

#print(timeit.timeit('st_average(fs)', globals=globals(), number=10))

#plt.step(fnp[0,:], fnp[1,:], where='post', linewidth=1)

favg2 = mem_average(fs)
print(timeit.timeit('matrix_l1_dist(fs)', globals=globals(), number=1))


fnp = favg2.to_numpy()
print(fnp.shape)
plt.step(fnp[0,:], fnp[1,:], '-.', where='post', linewidth=1)

plt.show()
