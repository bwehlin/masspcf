from mpcf.pcf import Pcf, average, mem_average, st_average, matrix_l1_dist, force_cpu, matrix_l1_dist_s
import numpy as np
import matplotlib.pyplot as plt
import timeit



m = 20000 # Number of PCFs
n = 100 # Number of time points in each PCF

#m = 32
#n = 5

T = np.random.uniform(0.0, 3.0, size=(m, n-1))
T = np.sort(T, axis=1)
T = np.hstack((T, 1.1 * np.max(np.max(T)) * np.ones((m, 1))))

V = np.sin(T) + 0.01 * np.random.randn(m, n)
V[:,-1] = 0

fs = [None]*m
for i in range(m):
    X = np.vstack((T[i,:], V[i,:]))
    fs[i] = Pcf(X)



#favg = average(fs)
#fnp = favg.to_numpy()
#print(fnp.shape)


#print(timeit.timeit('st_average(fs)', globals=globals(), number=10))

#plt.step(fnp[0,:], fnp[1,:], where='post', linewidth=1)

#favg2 = mem_average(fs)

print(f'Reqd int: {float(m)*float(m-1)/2.0}')

print('Running on CPU')
force_cpu(True)
cpu = matrix_l1_dist_s(fs)

print(cpu)
raise SystemExit
print('Running on GPU')
force_cpu(False)
gpu = matrix_l1_dist(fs)

print('Norm diff:')
print(np.linalg.norm(cpu - gpu))

print(cpu)
print(gpu)

