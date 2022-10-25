from collections.abc import Iterable
import numpy as np

from .snr import Snr

class QuantileSnr(Snr):
    """
    Estimate SNR as some high quantile of value/sigma. This is useful
    to measure the SNR in the continuum part of stellar spectra. 
    """

    def __init__(self, q=None, binning=1.0, orig=None):
        super().__init__(binning=binning, orig=orig)

        if not isinstance(orig, QuantileSnr):
            self.q = q if q is not None else 0.95
        else:
            self.q = q if q is not None else orig.q

    def get_snr(self, values, sigmas=None):

        if isinstance(values, np.ndarray):
            values = [ values ]

        if isinstance(sigmas, np.ndarray):
            sigmas = [ sigmas ]

        if sigmas is None:
            return np.nan
        else:
            v = np.concatenate(values)
            s = np.concatenate(sigmas)

            mask = (s > 0.0)
            snr = np.quantile(v[mask] / s[mask], self.q) * np.sqrt(self.binning)
            return snr

    