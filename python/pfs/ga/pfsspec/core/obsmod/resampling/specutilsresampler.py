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

    def resample_value(self, wave, wave_edges, value):
        # NOTE: SLOW!

        if value is None:
            return None
        else:
            # TODO: can we use wave_edges here?
            spec = Spectrum1D(spectral_axis=wave * u.AA, flux=value * u.Unit('erg cm-2 s-1 AA-1'))
            nspec = self.resampler(spec, self.wave * u.AA)
            return nspec.flux.value