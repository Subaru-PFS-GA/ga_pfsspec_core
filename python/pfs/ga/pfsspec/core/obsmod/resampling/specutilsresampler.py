from scipy.interpolate import interp1d
from astropy import units as u
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

from .resampler import Resampler

class SpecutilsResampler(Resampler):
    def __init__(self, resampler=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, SpecutilsResampler):
            self.resampler = resampler if resampler is not None else FluxConservingResampler()
        else:
            self.resampler = orig.resampler

    def resample_value(self, wave, wave_edges, value, error=None, target_wave=None, target_wave_edges=None):
        # NOTE: SLOW!

        # TODO: specutils version on the dev system is outdated,
        #       install at least 1.7 and try to get it calculate the error

        target_wave = target_wave if target_wave is not None else self.target_wave

        if value is None:
            ip_value = None
        else:
            # TODO: can we use wave_edges here?
            spec = Spectrum1D(spectral_axis=wave * u.AA, flux=value * u.Unit('erg cm-2 s-1 AA-1'))
            nspec = self.resampler(spec, target_wave * u.AA)
            ip_value = nspec.flux.value

        ip_error = None

        return ip_value, ip_error