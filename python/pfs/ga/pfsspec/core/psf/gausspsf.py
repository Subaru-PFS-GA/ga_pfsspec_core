import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..util.math import *
from .psf import Psf

class GaussPsf(Psf):
    """
    Implements a point spread function with a Gaussian kernel where sigma
    is tabulated as a function of wavelength.
    """

    def __init__(self, wave=None, wave_edges=None, sigma=None, orig=None):
        super().__init__(orig=orig)

        if isinstance(orig, GaussPsf):
            self.wave = wave or orig.wave
            self.wave_edges = wave_edges or orig.wave_edges
            self.sigma = sigma or orig.sigma
        else:
            self.wave = wave            # Wavelength grid on which sigma is defined
            self.wave_edges = wave_edges
            self.sigma = sigma          # Sigma of Gaussian PSF

        if self.wave is not None and self.sigma is not None:
            self.init_ip()
        else:
            self.ip = None

    def save_items(self):
        self.save_item('wave', self.wave)
        self.save_item('wave_edges', self.wave_edges)
        self.save_item('sigma', self.sigma)

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray, None)
        self.wave_edges = self.load_item('wave_edges', np.ndarray, None)
        self.sigma = self.load_item('sigma', np.ndarray, None)
        self.init_ip()

    def init_ip(self):
        # Interpolate sigma
        self.ip = interp1d(self.wave, self.sigma, bounds_error=False, fill_value=(self.wave[0], self.wave[-1]))

    def eval_kernel_at(self, lam, dwave, normalize=True):
        """
        Calculate the kernel around `lam` at `dwave` offsets.
        """
        sigma = self.ip(lam + dwave)
        k = gauss(dwave, s=sigma)

        if normalize:
            k = self.normalize(k)
        
        return k