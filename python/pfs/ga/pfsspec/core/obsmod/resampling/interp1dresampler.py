from scipy.interpolate import interp1d

from .resampler import Resampler

class Interp1dResampler(Resampler):
    def __init__(self, kind='linear', orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Interp1dResampler):
            self.kind = kind
        else:
            self.kind = orig.kind

    def resample_value(self, wave, wave_edges, value, error=None):
        if value is None:
            ip_value = None
        else:
            ip = interp1d(wave, value, kind=self.kind, bounds_error=True, assume_sorted=True)
            ip_value = ip(self.wave)

        if error is None:
            ip_error = None
        else:
            # For the error vector, use nearest-neighbor interpolations
            # later we can figure out how to do this correctly and add correlated noise, etc.

            # TODO: do this with correct propagation of error
            ip = interp1d(wave, error, kind='nearest', assume_sorted=True)
            ip_error = ip(self.wave)

        return ip_value, ip_error