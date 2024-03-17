import numpy as np
from .pcf import Pcf
from collections.abc import Callable

def noisy_function(func : Callable[[float], float] = lambda t : 0, n_times=25, noise=0.1, shape=(1,), dtype=np.float32):
    t = np.random.rand(n_times, 1)
    t.sort(axis=0)
    
    tv = np.hstack((t, np.zeros_like(t)))
    
    tv[0, 0] = 0
    tv[:, 1] = func(tv[:,0]) + noise * np.random.randn(n_times, )
    
    tv[-1, 1] = 0

    return Pcf(tv, dtype=dtype)