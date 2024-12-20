import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u

from .binning import Binning
from .resampler import Resampler

class FluxConservingResampler(Resampler):
    def __init__(self, kind=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, FluxConservingResampler):
            self.kind = kind if kind is not None else 'linear'
        else:
            self.kind = orig.kind

    def resample_flux(self, wave, wave_edges, flux, flux_err=None, mask=None,
                      target_wave=None, target_wave_edges=None):
        # TODO: How to properly resample using the flux conserving method
        #       when the source vector has masks?
        if mask is not None:
            raise NotImplementedError()
        
        target_wave = target_wave if target_wave is not None else self.target_wave
        target_wave_edges = target_wave_edges if target_wave_edges is not None else self.target_wave_edges

        # Find wave edges if they're not provided and make sure they're 1d arrays (no gaps between bins)
        # TODO: maybe we should allow for gaps between bins and use 2d edges array
        wave_edges = wave_edges if wave_edges is not None else self.find_wave_edges(wave)
        wave_edges = Binning.get_wave_edges_1d(wave_edges)
        target_wave_edges = target_wave_edges if target_wave_edges is not None else self.find_wave_edges(target_wave)
        target_wave_edges = Binning.get_wave_edges_1d(target_wave_edges)

        if flux is None:
            ip_value = None
            ip_mask = None
        else:
            # Mask out nan and inf values in input because those would compromise the results
            # TODO: combine this mask with the mask passed to the function
            mask = ~np.isnan(flux) & ~np.isinf(flux)
            mask_edges = np.empty_like(wave_edges, dtype=bool)
            mask_edges[0] = True
            mask_edges[1:] = mask
            
            # (Numerically) integrate the flux density as a function of wave at the upper
            # edges of the wavelength bins
            cs = np.empty_like(wave_edges[mask_edges], dtype=flux.dtype)
            cs[0] = 0.0
            cs[1:] = np.cumsum(flux[mask] * np.diff(wave_edges)[mask])
            ip = interp1d(wave_edges[mask_edges], cs, bounds_error=False, fill_value=(np.nan, np.nan), assume_sorted=True, kind=self.kind)
                        
            # Interpolate the integral and take the numerical differential
            if target_wave_edges.ndim == 1:
                ip_value = np.diff(ip(target_wave_edges)) / np.diff(target_wave_edges)
            elif target_wave_edges.ndim == 2:
                ip_value = (ip(target_wave_edges[1]) - ip(target_wave_edges[0])) / (target_wave_edges[1] - target_wave_edges[0])
            else:
                raise NotImplementedError()
            
            # Generate a mask if resampling outside original coverage
            ip_mask = ~np.isnan(ip_value) & (wave[0] <= target_wave_edges[:-1]) & (target_wave_edges[1:] <= wave[-1])
            
        # TODO: some time can be saved by only building the interpolator between the min and max of target_wave

        if flux_err is None:
            ip_error = None
        else:
            # For the error vector, use nearest-neighbor interpolations
            # later we can figure out how to do this correctly and add correlated noise, etc.

            # TODO: do this with correct propagation of error
            ip = interp1d(wave, flux_err, kind='nearest', assume_sorted=True)
            ip_error = ip(target_wave)

        return ip_value, ip_error, ip_mask