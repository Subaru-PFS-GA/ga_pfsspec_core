import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ...util.copy import safe_deep_copy
from ...util.math import *
from .psf import Psf

class GaussPsf(Psf):
    """
    Implements a point spread function with a Gaussian kernel where sigma
    is tabulated as a function of wavelength.
    """

    def __init__(self, wave=None, wave_edges=None, sigma=None, reuse_kernel=False, orig=None):
        super().__init__(reuse_kernel=reuse_kernel, orig=orig)

        if isinstance(orig, GaussPsf):
            self.wave = wave or safe_deep_copy(orig.wave)
            self.wave_edges = wave_edges or safe_deep_copy(orig.wave_edges)
            self.sigma = sigma or safe_deep_copy(orig.sigma)
        else:
            self.wave = wave            # Wavelength grid on which sigma is defined
            self.wave_edges = wave_edges
            self.sigma = sigma          # Sigma of Gaussian PSF

        if self.wave is not None and self.sigma is not None:
            self.init_ip()
        else:
            self.sigma_ip = None

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
        self.sigma_ip = interp1d(self.wave, self.sigma, bounds_error=False, fill_value=(self.sigma[0], self.sigma[-1]))

    @staticmethod
    def from_psf(source_psf, wave, dwave=None, size=None, s=slice(None), normalize=True, free_mean=True):
        """
        Evaluates the kernel as a function of wavelength on the provided grid
        and fits with a Gaussian function. While fitting, mean and amplitude can
        be used as free parameters but are not used when evaluating the kernel
        at arbitrary wavelengths.
        """

        w, dw, k, idx, shift = source_psf.eval_kernel(wave, dwave=dwave, size=size, s=s, normalize=normalize)

        # Fitting formula
        if not free_mean:
            g = lambda x, sig: gauss(x, m=0.0, s=sig, A=1.0)
            p0 = [1.5]          # default sigma
        else:
            g = lambda x, A, m, sig: gauss(x, m=m, s=sig, A=A)
            p0 = [1, 0, 1.5]    # A, m, s

        # Fit kernel at each wavelength
        sigma = np.zeros_like(w)
        for i in range(0, sigma.shape[0]):
            p, _ = curve_fit(g, dw[i, :], k[i, :], p0=p0)
            sigma[i] = p[-1]

        psf = GaussPsf()
        psf.wave = w
        psf.wave_edges = None
        psf.sigma = sigma
        psf.init_ip()

        return psf

    def eval_kernel_at(self, lam, dwave, normalize=True):
        """
        Calculate the kernel around `lam` at `dwave` offsets.
        """

        if not isinstance(lam, np.ndarray):
            lam = np.array([lam])

        sigma = self.sigma_ip(lam[:, np.newaxis] + dwave)
        k = gauss(dwave, s=sigma)

        if normalize:
            k = self.normalize(k)
        
        return k
    