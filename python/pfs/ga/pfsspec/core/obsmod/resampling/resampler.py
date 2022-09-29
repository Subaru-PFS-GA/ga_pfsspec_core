import numpy as np
from scipy.interpolate import interp1d

from pfs.ga.pfsspec.core.util.copy import *

class Resampler():
    def __init__(self, orig=None):
        
        if not isinstance(orig, Resampler):
            self.wave = None
            self.wave_edges = None
        else:
            self.wave = safe_deep_copy(orig.wave)
            self.wave_edges = safe_deep_copy(orig.wave_edges)

    def init(self, wave, wave_edges=None):
        self.wave = wave
        self.wave_edges = wave_edges if wave_edges is not None else self.find_wave_edges(wave)

    def reset(self):
        self.wave = None
        self.wave_edges = None

    def find_wave_edges(self, wave):
        # TODO: Do not automatically assume linear binning
        dw = wave[1:] - wave[:-1]
        wave_edges = np.empty((wave.shape[0] + 1,), dtype=wave.dtype)
        wave_edges[1:-1] = wave[1:] + 0.5 * dw
        wave_edges[0] = wave[0] - 0.5 * dw[0]
        wave_edges[-1] = wave[-1] + 0.5 * dw[-1]
        return wave_edges

    def find_centers(self, wave_edges):
        # TODO: Do not automatically assume linear binning
        return 0.5 * (wave_edges[1:] + wave_edges[:-1])

    def resample_value(self, wave, wave_edges, value, error=None):
        raise NotImplementedError()
   
    def resample_mask(self, wave, wave_edges, mask):
        # TODO: we only take the closest bin here which is incorrect
        #       mask values should be combined using bitwise or across bins

        if mask is None:
            return None
        else:
            wl_idx = np.digitize(self.wave, wave)
            return mask[wl_idx]