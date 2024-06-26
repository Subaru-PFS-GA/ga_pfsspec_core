from collections.abc import Iterable
import numpy as np

from .snr import Snr

class QuantileSnr(Snr):
    """
    Estimate SNR as some high quantile of value/sigma. This is useful
    to measure the SNR in the continuum part of stellar spectra. 
    """

    def __init__(self, q=None, binning=1.0, squared=None, orig=None):
        super().__init__(binning=binning, squared=squared, orig=orig)

        if not isinstance(orig, QuantileSnr):
            self.q = q if q is not None else 0.95
        else:
            self.q = q if q is not None else orig.q

    def get_snr_impl(self, value, sigma, mask):
        return np.quantile(value[mask] / sigma[mask], self.q) * np.sqrt(self.binning)

    