from scipy.interpolate import interp1d

from .resampler import Resampler

class Interp1dResampler(Resampler):
    def __init__(self, kind='linear', orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Interp1dResampler):
            self.kind = kind
        else:
            self.kind = orig.kind

    def resample_value(self, wave, wave_edges, value):
        if value is None:
            return None
        else:
            ip = interp1d(wave, value, kind=self.kind, bounds_error=True, assume_sorted=True)
            return ip(self.wave)
        