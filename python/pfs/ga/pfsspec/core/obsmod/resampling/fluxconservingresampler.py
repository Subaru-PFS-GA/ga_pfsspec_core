import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u

import pysynphot
from .resampler import Resampler

class FluxConservingResampler(Resampler):
    def __init__(self, resampler=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, FluxConservingResampler):
            pass
        else:
            pass

    def resample_value(self, wave, wave_edges, value, error=None):
        wave_edges = wave_edges if wave_edges is not None else self.find_wave_edges(wave)

        if value is None:
            ip_value = None
        else:
            # (Numerically) integrate the flux density as a function of wave at the upper
            # edges of the wavelength bins
            cs = np.cumsum(value * np.diff(wave_edges))
            ip = interp1d(wave_edges[1:], cs, bounds_error=False, fill_value=(0.0, np.nan))
            
            # Interpolate the integral and take the numerical differential
            ip_value = np.diff(ip(self.wave_edges)) / np.diff(self.wave_edges)

        if error is None:
            ip_error = None
        else:
            # For the error vector, use nearest-neighbor interpolations
            # later we can figure out how to do this correctly and add correlated noise, etc.

            # TODO: do this with correct propagation of error
            ip = interp1d(wave, error, kind='nearest', assume_sorted=True)
            ip_error = ip(self.wave)

        return ip_value, ip_error