#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:42:00 2024

@author: bwehlin
"""

import masspcf as mpcf
#import masspcf.system as system
from masspcf.plotting import plot as plotpcf
import numpy as np

import matplotlib.pyplot as plt

#mpcf.system.force_cpu(True)
mpcf.system.set_device_verbose(True)
mpcf.system.set_cuda_threshold(1)
mpcf.system.force_cpu(True)

f = mpcf.Pcf(np.array([[0, 4], [2, 3], [3, 1], [5, 0]]))
g = mpcf.Pcf(np.array([[0, 2], [6, 1], [7, 0]]))


print(mpcf.l2_kernel([f, g], verbose=False))

raise SystemExit

plotpcf(f)
plotpcf(g)
plt.show()

ps = np.linspace(1.0, 10.0)
ds = np.zeros_like(ps)
print(ds.shape)

for i, p in enumerate(ps):
    ds[i] = mpcf.pdist([f, g], p=p, verbose=False)[0,1]

plt.plot(ps, ds)

#plotpcf(f)