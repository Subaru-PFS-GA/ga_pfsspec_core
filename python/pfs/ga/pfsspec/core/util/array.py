import numpy as np

def copy_array(a):
    if a is None:
        return a
    else:
        return np.array(a, copy=True)