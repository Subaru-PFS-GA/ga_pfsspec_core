import numpy as np
from scipy.interpolate import interp1d

from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core.util.copy import *
from .binning import Binning

class Resampler(PfsObject):
    def __init__(self, target_wave=None, target_wave_edges=None, orig=None):
        super().__init__(orig=orig)
        
        if not isinstance(orig, Resampler):
            self.target_wave = target_wave
            self.target_wave_edges = target_wave_edges
        else:
            self.target_wave = safe_deep_copy(orig.target_wave)
            self.target_wave_edges = safe_deep_copy(orig.target_wave_edges)

    def init(self, target_wave, target_wave_edges=None):
        self.target_wave = target_wave
        self.target_wave_edges = target_wave_edges if target_wave_edges is not None else self.find_wave_edges(target_wave)

    def reset(self):
        self.target_wave = None
        self.target_wave_edges = None

    def find_wave_edges(self, wave):
        return Binning.find_wave_edges(wave)

    def find_wave_centers(self, wave_edges):
        # TODO: Do not automatically assume linear binning
        return 0.5 * (wave_edges[1:] + wave_edges[:-1])

    def resample_flux(self, wave, wave_edges, value, error=None, mask=None, target_wave=None, target_wave_edges=None):
        raise NotImplementedError()
   
    def resample_mask(self, wave, wave_edges, mask, target_wave=None, target_wave_edges=None):
        
        # If no mask is provided, return None.
        if mask is None:
            return None
        
        wave_edges = wave_edges if wave_edges is not None else self.find_wave_edges(wave)
        target_wave = target_wave if target_wave is not None else self.target_wave
        
        if target_wave_edges is not None:
            pass
        elif self.target_wave_edges is not None:
            target_wave_edges = self.target_wave_edges
        else:
            target_wave_edges = self.find_wave_edges(target_wave)

        # For each target wavelength pixel, find all corresponding pixels in the source wavelength
        # shape: (target_wave, wave)
        wave_idx = ((target_wave_edges[1:, ..., None] > wave_edges[:-1]) & (target_wave_edges[:-1, ..., None] < wave_edges[1:]))

        # Look up the mask for each source pixel and calculate the target mask
        # If the mask is boolean, the unmasked pixels are True so we take the logical AND of the source pixels
        # If the mask is integer, we calculate the bitwise OR of the source pixels
        if mask.dtype == bool:
            mask_idx = np.where(wave_idx, mask, False)
            target_mask = np.logical_and.reduce(mask_idx, axis=-1)
        else:
            mask_idx = np.where(wave_idx, mask, False)
            target_mask = np.bitwise_or.reduce(mask_idx, axis=-1)

        return target_mask

    def resample_weight(self, wave, wave_edges, weight, target_wave=None, target_wave_edges=None):
        target_wave = target_wave if target_wave is not None else self.target_wave

        # Weights are resampled using 1d interpolation.
        if weight is None:
            return None
        else:
            ip = interp1d(wave, weight)
            return ip(target_wave)
        