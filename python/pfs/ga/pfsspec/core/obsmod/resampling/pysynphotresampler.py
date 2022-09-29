import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u

import pysynphot
from .resampler import Resampler

class PysynphotResampler(Resampler):
    def __init__(self, resampler=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, PysynphotResampler):
            self.filt = None
        else:
            self.filt = orig.filt

    def init(self, wave, wave_edges=None):
        super().init(wave, wave_edges)

        # Assume an all-1 filter
        self.filt = pysynphot.spectrum.ArraySpectralElement(wave, np.ones(len(wave)), waveunits='angstrom')

    def reset(self):
        self.filt = None

    def resample_value(self, wave, wave_edges, value):
        # NOTE: SLOW!

        if value is None:
            return None
        else:
            # TODO: can we use wave_edges here?
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wave, flux=value, keepneg=True)
            obs = pysynphot.observation.Observation(spec, self.filt, binset=self.wave, force='taper')
            return obs.binflux

        