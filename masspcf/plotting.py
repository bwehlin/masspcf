import matplotlib.pyplot as plt
from .pcf import Pcf

def plot(f : Pcf, ax=None, **kwargs):
    ax = plt if ax is None else ax
    X = f.to_numpy()
    ax.step(X[0,:], X[1,:], **kwargs)
