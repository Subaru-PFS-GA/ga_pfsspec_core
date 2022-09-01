import numpy as np

def enumerate_axes(axes, squeeze=False):
    order = { k: axes[k].order for k in axes.keys() }
    keys = sorted(axes.keys(), key=lambda k: order[k])
    i = 0
    for k in keys:
        if not squeeze or axes[k].values.shape[0] > 1:
            yield i, k, axes[k]
            i += 1

def copy_array(a):
    if a is None:
        return a
    else:
        return np.array(a, copy=True)