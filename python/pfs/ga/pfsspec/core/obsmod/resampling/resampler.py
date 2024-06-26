import numpy as np
from scipy.interpolate import interp1d

from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core.util.copy import *
from .binning import Binning

class Resampler(PfsObject):
    def __init__(self, target_wave=None, target_wave_edges=None, target_mask=None, orig=None):
        super().__init__(orig=orig)
        
        if not isinstance(orig, Resampler):
            self.target_wave = target_wave
            self.target_wave_edges = target_wave_edges
            self.target_mask = target_mask
        else:
            self.target_wave = safe_deep_copy(orig.target_wave)
            self.target_wave_edges = safe_deep_copy(orig.target_wave_edges)
            self.target_mask = safe_deep_copy(orig.target_mask)

    def init(self, target_wave, target_wave_edges=None, target_mask=None):
        self.target_wave = target_wave
        self.target_wave_edges = target_wave_edges if target_wave_edges is not None else self.find_wave_edges(target_wave)
        self.target_mask = target_mask

    def reset(self):
        self.target_wave = None
        self.target_wave_edges = None
        self.target_mask = None

    def find_wave_edges(self, wave):
        return Binning.find_wave_edges(wave)

    def find_wave_centers(self, wave_edges):
        # TODO: Do not automatically assume linear binning
        return 0.5 * (wave_edges[1:] + wave_edges[:-1])

    def resample_flux(self, wave, wave_edges, value, error=None, mask=None, target_wave=None, target_wave_edges=None, target_mask=None):
        raise NotImplementedError()
   
    def resample_mask(self, wave, wave_edges, mask, target_wave=None, target_wave_edges=None):
        target_wave = target_wave if target_wave is not None else self.target_wave

        # TODO: we only take the closest bin here which is incorrect
        #       mask values should be combined using bitwise or across bins

        if mask is None:
            return None
        else:
            wl_idx = np.digitize(target_wave, wave)
            return mask[wl_idx]

    def resample_weight(self, wave, wave_edges, weight, target_wave=None, target_wave_edges=None):
        target_wave = target_wave if target_wave is not None else self.target_wave

        # Weights are resampled using 1d interpolation.
        if weight is None:
            return None
        else:
            ip = interp1d(wave, weight)
            return ip(target_wave)