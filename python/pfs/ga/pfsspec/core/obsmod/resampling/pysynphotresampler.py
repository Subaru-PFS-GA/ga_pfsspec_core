import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u

from ...setup_logger import logger
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

        if 'pysynphot' not in globals():
            try:
                import pysynphot
            except ModuleNotFoundError as ex:
                logger.warning(ex.msg)
                pysynphot = None

        # Assume an all-1 filter
        self.filt = pysynphot.spectrum.ArraySpectralElement(wave, np.ones(len(wave)), waveunits='angstrom')

    def reset(self):
        super().reset()

        self.filt = None

    def resample_flux(self, wave, wave_edges, value, error=None, mask=None, target_wave=None, target_wave_edges=None):
        # NOTE: SLOW!

        if 'pysynphot' not in globals():
            try:
                import pysynphot
            except ModuleNotFoundError as ex:
                logger.warning(ex.msg)
                pysynphot = None

        target_wave = target_wave if target_wave is not None else self.target_wave

        if value is None:
            ip_value = None
        else:
            # TODO: can we use wave_edges here?
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wave, flux=value, keepneg=True)
            obs = pysynphot.observation.Observation(spec, self.filt, binset=target_wave, force='taper')
            ip_value = obs.binflux

        # TODO: try to figure out how to handle error from pysynphot
        ip_error = None

        return ip_value, ip_error, None