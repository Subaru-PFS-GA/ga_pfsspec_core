import numpy as np

SQRT2PI = np.sqrt(2 * np.pi)

def gauss(x, m=0, s=1, A=1):
    return A / s / SQRT2PI * np.exp(-0.5 * ((x - m) / s)**2)