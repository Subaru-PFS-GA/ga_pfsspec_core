import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .psf import Psf

class LambdaPsf(Psf):
    """
    Implements a point spread function with an arbitrary kernel function provided
    as a callable.
    """

    def __init__(self, func, orig=None):
        super().__init__(orig=orig)

        if isinstance(orig, LambdaPsf):
            self.func = func
        else:
            self.func = func if func is not None else orig.func

    def get_kernel_at(self, lam, dwave, normalize=False):
        """
        Calculate the kernel around `lam` at `dwave` offsets.
        """
        k = self.func(lam, dwave)

        if normalize:
            k = self.normalize(k)

        return k