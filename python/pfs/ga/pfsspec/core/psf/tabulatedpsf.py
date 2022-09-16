from builtins import NotImplementedError, ValueError
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD

from .psf import Psf

class TabulatedPsf(Psf):
    """
    Computes the convolution of a data vector with a kernel tabulated as
    a function of wavelength.

    This kind of PSF representation uses a fixed kernel size with a fixed
    wavelength grid and fixed kernel wavelength steps.
    """

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, TabulatedPsf):
            self.wave = None            # wavelength grid
            self.wave_edges = None
            self.kernel = None
        else:
            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.kernel = orig.kernel

    def save_items(self):
        raise NotImplementedError()

    def load_items(self, s=None):
        raise NotImplementedError()

    def eval_kernel_impl(self, wave, size=None, s=slice(None), normalize=True):
        """
        Return the tabulated kernel.
        """

        # TODO: do we want to allow interpolation?

        if not np.array_equal(wave, self.wave):
            raise ValueError("Wave grid doesn't match tabulated grid.")

        if size is not None and size != self.kernel.shape[-1]:
            raise ValueError("Size doesn't match tabulated kernel size.")

        shift = self.kernel.shape[-1] // 2

        # idx will hold negative values because the tabulated kernels are usually
        # computed beyond the wave grid.
        idx = (np.arange(self.kernel.shape[-1]) - shift) + np.arange(wave.size)[:, np.newaxis]

        if normalize:
            k = self.normalize(self.kernel)
        else:
            k = self.kernel

        # Return 0 for shift since we have the kernel for the entire wavelength range
        return self.wave[s], k[s], idx[s], 0