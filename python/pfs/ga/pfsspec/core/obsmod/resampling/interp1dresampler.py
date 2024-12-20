import numpy as np
from scipy.interpolate import interp1d

from .resampler import Resampler

class Interp1dResampler(Resampler):
    def __init__(self, kind='linear', orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Interp1dResampler):
            self.kind = kind
        else:
            self.kind = orig.kind

    def resample_flux(self, wave, wave_edges, flux, flux_err=None, mask=None,
                      target_wave=None, target_wave_edges=None):
        
        target_wave = target_wave if target_wave is not None else self.target_wave

        if flux is None:
            ip_value = None
            ip_mask = None
        else:
            ip = interp1d(wave, flux, kind=self.kind, bounds_error=False, fill_value=(np.nan, np.nan), assume_sorted=True)
            ip_value = ip(target_wave)

            # Generate a mask if resampling outside original coverage
            ip_mask = ~np.isnan(ip_value)

        if flux_err is None:
            ip_error = None
        else:
            # For the error vector, use nearest-neighbor interpolations
            # later we can figure out how to do this correctly and add correlated noise, etc.

            # TODO: do this with correct propagation of error
            ip = interp1d(wave, flux_err, kind='nearest', bounds_error=False, fill_value=(np.nan, np.nan), assume_sorted=True)
            ip_error = ip(target_wave)

        return ip_value, ip_error, ip_mask