import matplotlib.pyplot as plt
from . import pcf

def plot(f : pcf.Pcf, ax=None, *args, **kwargs):
    if ax is None:
        ax = plt
    ax.step(f.to_numpy()[0,:], f.to_numpy()[1,:], *args, where='post', **kwargs)
