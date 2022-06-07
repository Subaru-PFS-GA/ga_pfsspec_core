import numpy as np

def enumerate_axes(axes):
    order = { k: axes[k].order for k in axes.keys() }
    keys = sorted(axes.keys(), key=lambda k: order[k])
    for i, k in enumerate(keys):
        yield i, k, axes[k]

def copy_array(a):
    if a is None:
        return a
    else:
        return np.array(a, copy=True)